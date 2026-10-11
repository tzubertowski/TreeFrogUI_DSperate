#include "core/cpu/arm_decode.h"
#include "core/cpu/cpu_cycles.h"
#include "core/cpu/jit/block_shape.h"
#include "core/cpu/jit/mips/convention.h"
#include "core/cpu/jit/jit_internal.h"
#include "core/cpu/jit/mips/emit.h"
#include "core/mem/timing.h"
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

extern "C" ds::u32 ds_jit_mips_fallback_block(ds::CpuContext *, const ds::u32 *,
                                              const ds::u32 *, ds::u32);
extern "C" ds::u32 jit_h_fallback(ds::CpuContext *, ds::u32, ds::u32);
extern "C" ds::u32 jit_h_ld8(ds::CpuContext *, ds::u32);
extern "C" ds::u32 jit_h_ld16(ds::CpuContext *, ds::u32);
extern "C" ds::u32 jit_h_ld32(ds::CpuContext *, ds::u32);
extern "C" void jit_h_st8(ds::CpuContext *, ds::u32, ds::u32);
extern "C" void jit_h_st16(ds::CpuContext *, ds::u32, ds::u32);
extern "C" void jit_h_st32(ds::CpuContext *, ds::u32, ds::u32);
extern "C" ds::u32 jit_h_mips_indirect_branch(ds::CpuContext *, ds::u32, ds::u32);
extern "C" ds::u32 jit_h_mips_exc_return(ds::CpuContext *, ds::u32);
extern "C" void jit_h_mips_set_cpsr(ds::CpuContext *, ds::u32);
namespace {
using ds::CpuContext;
using ds::s32;
using ds::u32;
using ds::arm::AOp;
using ds::jit::MipsEmitter;
constexpr u32 R_CTX = 16, R_SP = 29, R_FP = 30, R_RA = 31, R_BUDGET = 23;
constexpr u32 T0 = 8, T1 = 9, T2 = 10, T3 = 11, R_GUEST_PC = 24;
constexpr u32 OR = ds::jit::OFF_REGS, OB = ds::jit::OFF_BUDGET;
using ds::jit::CACHE_COUNT;
using ds::jit::CACHE_FIRST;
using ds::jit::CACHE_FRAME;
using ds::jit::CACHE_SAVED;
struct N {
  u32 op, rd, rn, rm, imm, shift, cond, rdlo, shift_reg;
  bool im, setflags, carry_valid, carry, carry_dynamic, reg_shift, rn_pc, rm_pc;
  bool signed_mul = false, accumulate = false, mul_arm7 = false;
  ds::u8 shift_type;
};
struct ThumbMem {
  u32 rd = 0, rn = 0, rm = 0, imm = 0, width = 0;
  u32 cond = 14, shift = 0, shift_type = 0, base = 0, offset = 0;
  bool reg = false, load = false, sign = false, arm = false;
  bool neg = false, base_const = false, offset_const = false;
  bool writeback = false, post = false;
};
static bool native_thumb_mem(u32 x, ThumbMem &m) {
  const u32 h = static_cast<ds::u16>(x);
  if ((h & 0xfe00u) == 0xb400u || (h & 0xfe00u) == 0xbc00u) {
    const bool pop = h & 0x0800u;
    const u32 list = h & 0xffu;
    const bool extra = h & 0x0100u;
    // shortcut: single-register PUSH/POP only, expand when block-transfer timing is native.
    if (__builtin_popcount(list) + extra != 1 || (pop && extra)) return false;
    m.rn = 13;
    m.rd = extra ? 14 : static_cast<u32>(__builtin_ctz(list));
    m.width = 4;
    m.load = pop;
    m.writeback = true;
    if (pop) {
      m.post = true;
      m.imm = 4;
    } else {
      m.neg = true;
      m.imm = 4;
    }
    return true;
  }
  if ((h & 0xfc00u) == 0x5000u) {
    static constexpr ds::u8 widths[8] = {4, 2, 1, 1, 4, 2, 1, 2};
    const u32 op = (h >> 9) & 7;
    m.width = widths[op];
    m.load = op >= 3;
    m.sign = op == 3 || op == 7;
    m.reg = true;
    m.rm = (h >> 6) & 7;
    m.rn = (h >> 3) & 7;
    m.rd = h & 7;
    return true;
  }
  if ((h & 0xe000u) == 0x6000u) {
    const bool byte = h & 0x1000u;
    m.width = byte ? 1 : 4;
    m.load = h & 0x0800u;
    m.rn = (h >> 3) & 7;
    m.rd = h & 7;
    m.imm = ((h >> 6) & 31) * m.width;
    return true;
  }
  if ((h & 0xf000u) == 0x8000u) {
    m.width = 2;
    m.load = h & 0x0800u;
    m.rn = (h >> 3) & 7;
    m.rd = h & 7;
    m.imm = ((h >> 6) & 31) * 2;
    return true;
  }
  if ((h & 0xf000u) == 0x9000u) {
    m.width = 4;
    m.load = h & 0x0800u;
    m.rn = 13;
    m.rd = (h >> 8) & 7;
    m.imm = (h & 255) * 4;
    return true;
  }
  return false;
}
static bool native_arm_mem(u32 x, u32 pc, ThumbMem &m, bool arm9) {
  const AOp op = ds::arm::decode_arm(x);
  const bool extra = op == AOp::LdrStrHImm || op == AOp::LdrStrHReg;
  if ((!extra && op != AOp::LdrStrImm && op != AOp::LdrStrReg) ||
      (x & (1u << 21))) return false; // unprivileged post-index is not covered
  m.arm = true;
  m.cond = x >> 28;
  m.load = x & (1u << 20);
  const u32 sh = (x >> 5) & 3;
  m.width = extra ? (sh == 2 ? 1 : 2) : (x & (1u << 22)) ? 1 : 4;
  m.sign = extra && m.load && sh != 1;
  m.neg = !(x & (1u << 23));
  m.post = !(x & (1u << 24));
  m.writeback = m.post;
  m.rn = (x >> 16) & 15;
  m.rd = (x >> 12) & 15;
  m.base_const = m.rn == 15;
  // ARM7 register-base PC loads may read pipeline-dependent open bus values.
  if ((m.rd == 15 && (!m.load || m.width != 4 || extra || (!arm9 && m.rn != 15))) ||
      m.cond == 15 || (m.base_const && m.post) ||
      (m.writeback && m.load && m.rd == m.rn) ||
      (extra && ((!m.load && sh != 1) || sh == 0)))
    return false; // LDRD/STRD are intentionally left to the interpreter.
  m.base = pc + 8;
  if (op == AOp::LdrStrImm || op == AOp::LdrStrHImm) {
    m.imm = extra ? (((x >> 4) & 0xf0u) | (x & 15u)) : (x & 0xfffu);
    return true;
  }
  if (!extra && (x & (1u << 4))) return false; // register-specified shift
  m.reg = true;
  m.rm = x & 15;
  m.offset_const = m.rm == 15;
  m.offset = pc + 8;
  if (!extra) {
    m.shift_type = (x >> 5) & 3;
    m.shift = (x >> 7) & 31;
    if (m.shift_type == 3 && !m.shift) return false; // RRX depends on CPSR.C
  }
  return true;
}
static bool native_arm_single_multi(u32 x, ThumbMem &m) {
  const AOp op = ds::arm::decode_arm(x);
  const u32 list = x & 0xffffu;
  if ((op != AOp::Ldm && op != AOp::Stm) || !list || (list & (list - 1)) ||
      (x >> 28) == 15 || ((x >> 16) & 15) == 15)
    return false;
  const u32 rd = static_cast<u32>(__builtin_ctz(list));
  const u32 rn = (x >> 16) & 15;
  if (rd == 15 || (x & (1u << 22)) ||
      ((x & (1u << 21)) && rd == rn)) return false;
  m.arm = true;
  m.cond = x >> 28;
  m.load = x & (1u << 20);
  m.width = 4;
  m.rn = rn;
  m.rd = rd;
  m.neg = !(x & (1u << 23));
  m.imm = 4;
  m.post = !(x & (1u << 24));
  m.writeback = x & (1u << 21);
  return true;
}
static bool native_arm_block(u32 x) {
  const AOp op = ds::arm::decode_arm(x);
  const u32 list = x & 0xffffu;
  const u32 rn = (x >> 16) & 15;
  return (op == AOp::Ldm || op == AOp::Stm) && __builtin_popcount(list) > 1 &&
         (x >> 28) != 15 && !(x & (1u << 22)) && rn != 15 &&
         !(list & (1u << 15)) && !(list & (1u << rn));
}
static int arm7_data_slot(const CpuContext &c, u32 pc, bool thumb,
                          u32 width, bool load) {
  const u32 nc = c.timing7[pc >> 15][thumb ? 0 : 2];
  const int ni = c.nds->bus.timing().nc7_index(nc);
  return ni < 0 ? -1 : static_cast<int>(ds::mem::Timing::cost7_offset(
      (pc >> 24) == 2, load, static_cast<u32>(ni), width == 4));
}
static u32 ai(u32 x) {
  u32 v = x & 255, n = ((x >> 8) & 15) * 2;
  return n ? (v >> n) | (v << (32 - n)) : v;
}
static bool native_op_enabled(u32 op, bool arm9) {
  const char *env = !arm9 ? std::getenv("DS_MIPS_NATIVE_OP_ARM7") : nullptr;
  if (!env) env = std::getenv("DS_MIPS_NATIVE_OP");
  if (!env || !std::strcmp(env, "all")) return true;
  for (const char *p = env; *p;) {
    char *end = nullptr;
    const unsigned long value = std::strtoul(p, &end, 10);
    if (end == p) return false;
    if (value == op) return true;
    if (*end != ',') return false;
    p = end + 1;
  }
  return false;
}
static bool native(u32 x, N &n, bool shifts, bool arm9) {
  AOp a = ds::arm::decode_arm(x);
  n.cond = x >> 28;
  if (n.cond == 15 || (a != AOp::DpImm && a != AOp::DpImmShift &&
                       a != AOp::DpRegShift))
    return false;
  n.op = (x >> 21) & 15;
  // "all" enables audited ALU forms; a numeric value keeps one-op canaries
  // available for hardware where a particular generated opcode is suspect.
  if (!native_op_enabled(n.op, arm9)) return false;
  n.rd = (x >> 12) & 15;
  n.rn = (x >> 16) & 15;
  n.rm = x & 15;
  n.setflags = x & (1u << 20);
  // In immediate forms the low nibble belongs to imm8, not Rm.  Rejecting
  // rm==15 here silently routed valid immediate ALU instructions to fallback.
  if (n.rd == 15 || (a == AOp::DpRegShift && n.rm == 15))
    return false;
  n.rn_pc = n.rn == 15;
  n.rm_pc = a == AOp::DpImmShift && n.rm == 15;
  if (a == AOp::DpImm && n.op == 15 && !n.setflags && ai(x) == 0) {
    n.op = 13;
    n.imm = ~0u;
    n.im = true;
    return true;
  }
  // BIC/MVN use arithmetic complements here; do not emit the failing NOR form.
  if (n.op != 0 && n.op != 1 && n.op != 2 && n.op != 3 && n.op != 4 &&
      n.op != 5 && n.op != 6 && n.op != 7 && n.op != 8 && n.op != 9 &&
      n.op != 10 && n.op != 11 && n.op != 12 && n.op != 13 &&
      n.op != 14 && n.op != 15)
    return false;
  const bool test_op = n.op == 8 || n.op == 9 || n.op == 10 || n.op == 11;
  const bool logical_flags = n.op == 0 || n.op == 1 || n.op == 8 || n.op == 9 ||
                             n.op == 12 || n.op == 13 || n.op == 14 || n.op == 15;
  if (test_op && !n.setflags)
    return false;
  if (n.setflags && !logical_flags && n.op != 2 && n.op != 3 && n.op != 4 &&
      n.op != 5 && n.op != 6 && n.op != 7 &&
      n.op != 10 && n.op != 11)
    return false;
  if (a == AOp::DpImm) {
    n.im = true;
    n.imm = ai(x);
    n.carry_valid = ((x >> 8) & 15) != 0;
    n.carry = n.imm >> 31;
    return true;
  }
  if (a == AOp::DpRegShift) {
    n.reg_shift = true;
    n.shift_reg = (x >> 8) & 15;
    if (n.shift_reg == 15) return false;
    n.shift_type = (x >> 5) & 3;
    n.carry_dynamic = n.setflags &&
        (n.op == 0 || n.op == 1 || n.op == 8 || n.op == 9 || n.op >= 12);
    n.carry_valid = n.carry_dynamic;
    return shifts;
  }
  if ((x & 0x10) || (!shifts && ((x >> 4) & 255)))
    return false;
  n.im = false;
  n.shift = (x >> 7) & 31;
  n.shift_type = (x >> 5) & 3;
  n.carry_dynamic = n.setflags && (n.shift || n.shift_type);
  n.carry_valid = n.carry_dynamic;
  return n.shift_type == 0 ||
         (!n.shift && (n.shift_type == 1 || n.shift_type == 2)) ||
         (!n.shift && n.shift_type == 3) ||
         (n.shift && n.shift_type == 3) ||
         (n.shift && (n.shift_type == 1 || n.shift_type == 2));
}
static bool native_thumb(u32 x, N &n, bool arm9) {
  const ds::u16 h = static_cast<ds::u16>(x);
  const bool mov_imm = (h & 0xf800) == 0x2000;
  const bool imm_shift = (h & 0xe000) == 0 && ((h >> 11) & 3) != 3;
  const bool imm8_arith = (h & 0xf800) == 0x3000 || (h & 0xf800) == 0x3800;
  const bool cmp_imm8 = (h & 0xf800) == 0x2800;
  const bool imm3_arith = (h & 0xfe00) == 0x1c00 || (h & 0xfe00) == 0x1e00;
  const bool reg3_arith = (h & 0xfe00) == 0x1800 || (h & 0xfe00) == 0x1a00;
  const bool high_op = (h & 0xfc00) == 0x4400 && ((h >> 8) & 3) < 3;
  const u32 high_kind = (h >> 8) & 3;
  const bool high_mov = high_op && high_kind == 2;
  const bool high_add = high_op && high_kind == 0;
  const bool high_cmp = high_op && high_kind == 1;
  const bool alu_reg = (h & 0xfc00) == 0x4000;
  const u32 alu_op = (h >> 6) & 15;
  if (!mov_imm && !imm_shift && !imm8_arith && !cmp_imm8 && !imm3_arith && !alu_reg &&
      !reg3_arith && !high_mov && !high_add && !high_cmp)
    return false;
  if (mov_imm || imm_shift || high_mov) n.op = 13;
  else if (high_add) n.op = 4;
  else if (high_cmp) n.op = 10;
  else if (cmp_imm8) n.op = 10;
  else if (imm8_arith || imm3_arith || reg3_arith)
    n.op = (h & (imm8_arith ? 0x0800 : 0x0200)) ? 2 : 4;
  else n.op = alu_op;
  if (!native_op_enabled(n.op, arm9)) return false;
  n.setflags = !high_mov && !high_add;
  n.cond = 14;
  if (mov_imm) {
    n.rd = (h >> 8) & 7;
    n.imm = h & 255;
    n.im = true;
  } else if (imm_shift) {
    n.rd = h & 7;
    n.rm = (h >> 3) & 7;
    n.shift = (h >> 6) & 31;
    n.shift_type = (h >> 11) & 3;
    n.carry_dynamic = n.shift_type != 0;
    n.carry_valid = n.carry_dynamic;
  } else if (imm8_arith || cmp_imm8) {
    n.rd = (h >> 8) & 7;
    n.rn = n.rd;
    n.imm = h & 255;
    n.im = true;
  } else if (imm3_arith) {
    n.rd = h & 7;
    n.rn = (h >> 3) & 7;
    n.imm = (h >> 6) & 7;
    n.im = true;
  } else if (reg3_arith) {
    n.rd = h & 7;
    n.rn = (h >> 3) & 7;
    n.rm = (h >> 6) & 7;
  } else if (alu_reg) {
    n.rd = h & 7;
    n.rn = n.rd;
    n.rm = (h >> 3) & 7;
    if ((alu_op >= 2 && alu_op <= 4) || alu_op == 7) {
      n.op = 13;
      n.reg_shift = true;
      n.shift_type = alu_op == 2 ? 0 : alu_op == 3 ? 1 : alu_op == 4 ? 2 : 3;
      n.shift_reg = n.rm;
      n.rm = n.rd;
      n.carry_dynamic = n.carry_valid = true;
    } else if (alu_op == 9) {
      n.op = 3;
      n.rn = n.rm;
      n.im = true;
      n.imm = 0;
    } else if (alu_op == 13) {
      n.op = 16;
      n.rn = n.rd;
    } else if (alu_op == 8 || alu_op == 10 || alu_op == 11) {
      n.rd = 15;
    }
  } else if (high_mov || high_add || high_cmp) {
    n.rd = (h & 7) | ((h >> 4) & 8);
    n.rm = (h >> 3) & 15;
    n.rn = n.rd;
    if (n.rd == 15 || n.rm == 15) return false;
    if (high_cmp) n.rd = 15;
  }
  return true;
}
static bool native_insn(u32 x, bool thumb, bool arm9, N &n) {
  const AOp decoded = !thumb ? ds::arm::decode_arm(x) : AOp::Undefined;
  if (!thumb && arm9 && (decoded == AOp::Umull || decoded == AOp::Umlal ||
                         decoded == AOp::Smull || decoded == AOp::Smlal)) {
    const u32 hi = (x >> 16) & 15u, lo = (x >> 12) & 15u;
    const u32 rs = (x >> 8) & 15u, rm = x & 15u;
    // shortcut: overlapping long multiplies use the interpreter after a ROM
    // mismatch; remove once cache aliasing is understood.
    if (hi == 15 || lo == 15 || rs == 15 || rm == 15 || hi == lo ||
        hi == rs || hi == rm || lo == rs || lo == rm)
      return false;
    if (!native_op_enabled(17, arm9)) return false;
    n.op = 17; n.rd = hi; n.rdlo = lo; n.rn = rs; n.rm = rm; n.cond = x >> 28;
    n.setflags = x & (1u << 20);
    n.signed_mul = decoded == AOp::Smull || decoded == AOp::Smlal;
    n.accumulate = decoded == AOp::Umlal || decoded == AOp::Smlal;
    return n.cond != 15;
  }
  if (!thumb && (decoded == AOp::Mul || decoded == AOp::Mla)) {
    const bool acc = decoded == AOp::Mla;
    const u32 rd = (x >> 16) & 15u, rn = (x >> 12) & 15u;
    const u32 rs = (x >> 8) & 15u, rm = x & 15u;
    if ((!acc && rn) || rd == 15 || (acc && rn == 15) || rs == 15 || rm == 15 ||
        (x >> 28) == 15)
      return false;
    if (!native_op_enabled(16, arm9)) return false;
    n.op = 16;
    n.rd = rd;
    n.rn = rs;
    n.rm = rm;
    n.rdlo = acc ? rn : 0;
    n.accumulate = acc;
    n.mul_arm7 = !arm9;
    n.setflags = x & (1u << 20);
    n.cond = x >> 28;
    return n.rd != 15;
  }
  if (!thumb) return native(x, n, true, arm9);
  if (!native_thumb(x, n, arm9)) return false;
  n.mul_arm7 = !arm9 && n.op == 16;
  return true;
}
static void emit_skip_if_false(MipsEmitter &e, u32 cond, std::vector<size_t> &skip) {
  if (cond == 14) return;
  e.lw(T0, ds::jit::OFF_CPSR, R_CTX);
  e.nop();
  auto flag = [&](u32 bit) { e.srl(T1, T0, bit); e.andi(T1, T1, 1); };
  auto fail_beq = [&](u32 a, u32 b) { skip.push_back(e.beq(a, b)); e.nop(); };
  auto fail_bne = [&](u32 a, u32 b) { skip.push_back(e.bne(a, b)); e.nop(); };
  auto fail_bnez = [&](u32 a) { skip.push_back(e.bnez(a)); e.nop(); };
  auto fail_beqz = [&](u32 a) { fail_beq(a, 0); };
  auto nv_equal = [&] {
    e.srl(T1, T0, 31); e.andi(T1, T1, 1);
    e.srl(T2, T0, 28); e.andi(T2, T2, 1);
  };
  switch (cond) {
  case 0: flag(30); fail_beqz(T1); break; // EQ
  case 1: flag(30); fail_bnez(T1); break;
  case 2: flag(29); fail_beqz(T1); break; // CS
  case 3: flag(29); fail_bnez(T1); break;
  case 4: flag(31); fail_beqz(T1); break; // MI
  case 5: flag(31); fail_bnez(T1); break;
  case 6: flag(28); fail_beqz(T1); break; // VS
  case 7: flag(28); fail_bnez(T1); break;
  case 8: flag(29); fail_beqz(T1); flag(30); fail_bnez(T1); break; // HI
  case 9: {
    flag(29);
    const size_t c_clear = e.beq(T1, 0);
    e.nop();
    flag(30);
    fail_beqz(T1);
    e.patch_branch(c_clear, e.size());
    break;
  } // LS
  case 10: nv_equal(); fail_bne(T1, T2); break; // GE
  case 11: nv_equal(); fail_beq(T1, T2); break;
  case 12: flag(30); fail_bnez(T1); nv_equal(); fail_bne(T1, T2); break; // GT
  case 13: {
    flag(30);
    const size_t z_set = e.bnez(T1);
    e.nop();
    nv_equal();
    fail_beq(T1, T2);
    e.patch_branch(z_set, e.size());
    break;
  }
  }
}
static void li(MipsEmitter &e, u32 r, u32 v) {
  if (v < 0x8000u) {
    e.addiu(r, 0, static_cast<s32>(v));
    return;
  }
  e.lui(r, v >> 16);
  e.ori(r, r, v);
}
static u32 arm9_code_cost(const CpuContext &c, u32 pc, bool thumb) {
  const u32 fetch_pc = pc + (thumb ? 4 : 8);
  const ds::u8 t = c.timing9[fetch_pc >> 12][0];
  return thumb && (fetch_pc & 2u) ? 0 :
         t == 255 ? (!(fetch_pc & 31u) ? 3 : 1) : t;
}
static void emit_arm9_data_extra(MipsEmitter &e, const CpuContext &c, u32 pc,
                                 bool thumb, u32 width, bool store) {
  const u32 nc = arm9_code_cost(c, pc, thumb);
  const u32 slot = (store ? 4 : 0) + (width == 4 ? 2 : 1);
  e.lw(T2, offsetof(CpuContext, timing9), R_CTX);
  e.nop();
  if (width == 4) {
    e.srl(T3, R_FP, 2);
    e.sll(T3, T3, 2);
  } else if (width == 2) {
    e.srl(T3, R_FP, 1);
    e.sll(T3, T3, 1);
  } else {
    e.move(T3, R_FP);
  }
  e.srl(T3, T3, 12);
  e.sll(T3, T3, 3);
  e.addu(T2, T2, T3);
  e.lbu(T2, slot, T2);
  e.nop();
  if (nc <= 6) {
    if (nc) {
      e.addiu(T1, T2, -static_cast<s32>(nc));
      li(e, T0, nc);
      e.sltu(T3, T2, T0);
      e.xori(T3, T3, 1);
      e.subu(T3, 0, T3);
      e.and_(T2, T1, T3);
    }
  } else {
    e.addiu(T1, T2, -6);
    e.addiu(T0, 0, 6);
    e.sltu(T3, T2, T0);
    e.xori(T3, T3, 1);
    e.subu(T3, 0, T3);
    e.and_(T2, T1, T3);
  }
  e.subu(R_BUDGET, R_BUDGET, T2);
}
static void emit_data_extra(MipsEmitter &e, const CpuContext &c, u32 pc,
                            bool thumb, u32 width, bool store, int slot7) {
  if (c.which == ds::Cpu::ARM9) {
    emit_arm9_data_extra(e, c, pc, thumb, width, store);
    return;
  }
  const u32 seq = c.timing7[pc >> 15][thumb ? 1 : 3];
  e.lw(T2, offsetof(CpuContext, cost7), R_CTX);
  e.srl(T3, R_FP, 15);
  e.sll(T3, T3, 5);
  e.addu(T2, T2, T3);
  e.lbu(T2, slot7, T2);
  e.nop();
  if (seq) e.addiu(T2, T2, -static_cast<s32>(seq));
  e.subu(R_BUDGET, R_BUDGET, T2);
}
static void getr(MipsEmitter &e, u32 d, u32 r, u32 cache_mask);
static void putr(MipsEmitter &e, u32 r, u32 s, u32 cache_mask);
static void spill_cached(MipsEmitter &e, u32 cache_mask);
static void reload_cached(MipsEmitter &e, u32 cache_mask);
static void emit_thumb_mem(MipsEmitter &e, const CpuContext &c, u32 pc,
                           bool thumb, const ThumbMem &m, u32 cache_mask,
                           std::vector<size_t> &ex, int slot7) {
  const bool pc_dest = m.arm && m.rd == 15;
  // Tagged/MMIO/write-protected pages retain helper semantics, including SMC
  // and device IO; plain mapped RAM uses the direct path on either CPU.
  if (m.base_const) li(e, T0, m.base);
  else getr(e, T0, m.rn, cache_mask);
  bool have_offset = false;
  if (m.reg) {
    if (m.offset_const) li(e, T1, m.offset);
    else getr(e, T1, m.rm, cache_mask);
    if (m.arm && (m.shift || m.shift_type != 0)) {
      if (m.shift_type == 0) e.sll(T1, T1, m.shift);
      else if (m.shift_type == 1) {
        if (m.shift) e.srl(T1, T1, m.shift);
        else e.move(T1, 0); // ARM LSR #0 means LSR #32.
      }
      else if (m.shift_type == 2) e.sra(T1, T1, m.shift ? m.shift : 31);
      else e.rotr(T1, T1, m.shift);
    }
    have_offset = true;
  } else if (m.arm) {
    li(e, T1, m.imm);
    have_offset = true;
  }
  if (m.post) {
    if (!have_offset) li(e, T1, m.imm);
    if (m.neg) e.subu(T3, T0, T1);
    else e.addu(T3, T0, T1);
    e.sw(T3, 52, R_SP);
  } else if (m.arm && have_offset) {
    if (m.neg) e.subu(T0, T0, T1);
    else e.addu(T0, T0, T1);
    if (m.writeback) e.sw(T0, 52, R_SP);
  } else if (m.reg) {
    if (m.neg) e.subu(T0, T0, T1);
    else e.addu(T0, T0, T1);
  } else if (m.imm) {
    e.addiu(T0, T0, m.neg ? -static_cast<s32>(m.imm) : static_cast<s32>(m.imm));
    if (m.writeback) e.sw(T0, 52, R_SP);
  }
  e.move(R_FP, T0);                    // original EA survives helper calls
  if (m.width == 4) { e.srl(T0, T0, 2); e.sll(T0, T0, 2); }
  else if (m.width == 2) { e.srl(T0, T0, 1); e.sll(T0, T0, 1); }
  e.srl(T1, T0, 11);                  // page-table index
  e.sll(T1, T1, 2);
  e.load_ptr(T2, c.page_table.raw());
  e.addu(T2, T2, T1);
  e.lw(T1, 0, T2);
  e.nop();
  if (m.load) {
    e.srl(T2, T1, 30); e.andi(T2, T2, 1); // SPECIAL pages need helper
  } else {
    e.srl(T2, T1, 30);                    // CODE/SPECIAL stores need helper
  }
  const size_t tagged = e.bnez(T2); e.nop();
  e.sll(T2, T1, 2);                       // remove both tag bits
  const size_t unmapped = e.beq(T2, 0); e.nop();
  e.addu(T2, T2, T0);                     // mapped host address
  if (m.load) {
    if (m.width == 1) { e.lbu(T1, 0, T2); e.nop(); }
    else if (m.width == 2) { e.lhu(T1, 0, T2); e.nop(); }
    else { e.lw(T1, 0, T2); e.nop(); }
  } else {
    getr(e, T3, m.rd, cache_mask);
    if (m.width == 1) e.sb(T3, 0, T2);
    else if (m.width == 2) e.sh(T3, 0, T2);
    else e.sw(T3, 0, T2);
  }
  const size_t fast_done = e.b(); e.nop();
  const size_t slow = e.size();
  e.patch_branch(tagged, slow);
  e.patch_branch(unmapped, slow);
  auto loaded = [&] {
    if (!m.load) return;
    if (m.width == 4) {
      e.andi(T3, R_FP, 3); e.sll(T3, T3, 3);
      e.move(T0, T1); e.srlv(T1, T0, T3);
      e.subu(T3, 0, T3); e.sllv(T2, T0, T3); e.or_(T1, T1, T2);
    } else if (m.sign) {
      e.sll(T1, T1, m.width == 1 ? 24 : 16);
      e.sra(T1, T1, m.width == 1 ? 24 : 16);
    }
    if (!pc_dest) putr(e, m.rd, T1, cache_mask);
  };
  auto writeback = [&] {
    if (!m.writeback) return;
    e.lw(T3, 52, R_SP); e.nop();
    putr(e, m.rn, T3, cache_mask);
  };
  auto branch_pc = [&] {
    e.sw(T1, 60, R_SP);
    emit_data_extra(e, c, pc, thumb, m.width, false, slot7);
    e.lw(T1, 60, R_SP); e.nop();
    e.move(4, R_CTX); e.move(5, T1);
    e.addiu(6, 0, c.which == ds::Cpu::ARM9 ? 1 : 0);
    e.load_ptr(25, reinterpret_cast<const void *>(&jit_h_mips_indirect_branch));
    spill_cached(e, cache_mask);
    e.sw(R_GUEST_PC, OR + 60, R_CTX); e.sw(R_BUDGET, OB, R_CTX);
    e.jalr(R_RA, 25); e.nop();
    e.move(R_GUEST_PC, 2);
    e.lw(R_BUDGET, OB, R_CTX); e.nop();
    reload_cached(e, cache_mask);
    ex.push_back(e.b()); e.nop();
  };
  e.move(4, R_CTX);
  e.move(5, R_FP);
  if (m.load) {
    const void *fn = m.width == 1 ? reinterpret_cast<const void *>(&jit_h_ld8) :
                     m.width == 2 ? reinterpret_cast<const void *>(&jit_h_ld16) :
                                    reinterpret_cast<const void *>(&jit_h_ld32);
    e.load_ptr(25, fn);
  } else {
    getr(e, 6, m.rd, cache_mask);
    const void *fn = m.width == 1 ? reinterpret_cast<const void *>(&jit_h_st8) :
                     m.width == 2 ? reinterpret_cast<const void *>(&jit_h_st16) :
                                    reinterpret_cast<const void *>(&jit_h_st32);
    e.load_ptr(25, fn);
  }
  spill_cached(e, cache_mask);
  e.sw(R_GUEST_PC, OR + 60, R_CTX);
  e.sw(R_BUDGET, OB, R_CTX);
  e.jalr(R_RA, 25); e.nop();
  if (m.load) e.move(T1, 2);
  e.lw(R_BUDGET, OB, R_CTX); e.nop();
  e.lw(R_GUEST_PC, OR + 60, R_CTX); e.nop();
  reload_cached(e, cache_mask);
  loaded();
  writeback();
  if (pc_dest) branch_pc();
  else {
    emit_data_extra(e, c, pc, thumb, m.width, !m.load, slot7);
    e.addiu(R_GUEST_PC, R_GUEST_PC, thumb ? 2 : 4);
    ex.push_back(e.blez(R_BUDGET)); e.nop();
    // Slow operations may have raised scheduler alerts; return at this boundary.
    ex.push_back(e.b()); e.nop();
  }
  e.patch_branch(fast_done, e.size());
  loaded();
  writeback();
  if (pc_dest) branch_pc();
  else {
    emit_data_extra(e, c, pc, thumb, m.width, !m.load, slot7);
    e.addiu(R_GUEST_PC, R_GUEST_PC, thumb ? 2 : 4);
    ex.push_back(e.blez(R_BUDGET)); e.nop();
  }
  // Dynamic timing tables mean a timing update can affect any page used here.
}
static void emit_timing(MipsEmitter &e, const CpuContext &c, u32 pc, bool thumb,
                        bool internal = false);
static void emit_arm7_block_cost(MipsEmitter &e, const CpuContext &c, u32 pc,
                                 u32 start, bool load, u32 n) {
  e.move(T0, 0);
  e.lw(T3, offsetof(CpuContext, timing7), R_CTX); e.nop();
  for (u32 k = 0; k < n; ++k) {
    if (k) e.addiu(T1, start, static_cast<s32>(k * 4));
    else e.move(T1, start);
    e.srl(T1, T1, 15); e.sll(T1, T1, 2); e.addu(T1, T3, T1);
    e.lbu(T1, k ? 3 : 2, T1); e.nop(); e.addu(T0, T0, T1);
  }
  const u32 nc = c.timing7[pc >> 15][2];
  auto max = [&] {
    e.sltu(T3, T1, T0); e.subu(T3, 0, T3);
    e.xor_(T0, T0, T1); e.and_(T0, T0, T3); e.xor_(T0, T0, T1);
  };
  auto different_region_cost = [&] {
    e.addu(T2, T0, T1);
    max();
    e.sltiu(T3, T2, 3);
    const size_t no_combine = e.bnez(T3); e.nop();
    e.addiu(T2, T2, -3);
    e.move(T1, T2); max();
    e.patch_branch(no_combine, e.size());
  };
  const bool code_main = (pc >> 24) == 2;
  e.srl(T2, start, 24); e.addiu(T3, 0, 2);
  const size_t data_main = e.beq(T2, T3); e.nop();
  if (code_main) {
    e.addiu(T1, 0, nc);
    if (load) e.addiu(T0, T0, 1);
    different_region_cost();
  } else if (load) {
    e.addiu(T1, 0, nc + 1);
    e.addu(T0, T0, T1);
  } else {
    e.addiu(T1, 0, nc);
    different_region_cost();
  }
  const size_t cross_done = e.b(); e.nop();
  e.patch_branch(data_main, e.size());
  if (code_main) {
    e.addiu(T1, 0, nc);
    e.addu(T0, T0, T1);
  } else {
    e.addiu(T1, 0, nc);
    if (load) e.addiu(T1, T1, 1); // ARM7's load internal cycle belongs to code.
    different_region_cost();
  }
  e.patch_branch(cross_done, e.size());
  e.subu(R_BUDGET, R_BUDGET, T0);
}
static void emit_arm_block_mem(MipsEmitter &e, const CpuContext &c, u32 pc,
                               u32 instr, u32 cache_mask,
                               std::vector<size_t> &ex) {
  const u32 list = instr & 0xffffu, rn = (instr >> 16) & 15;
  const u32 n = __builtin_popcount(list);
  const bool load = (instr & (1u << 20)) != 0;
  const bool up = (instr & (1u << 23)) != 0;
  const bool pre = (instr & (1u << 24)) != 0;
  const bool writeback = (instr & (1u << 21)) != 0;
  std::vector<size_t> not_executed, slow, done;
  emit_skip_if_false(e, instr >> 28, not_executed);

  getr(e, T0, rn, cache_mask);
  if (up) e.addiu(R_FP, T0, pre ? 4 : 0);
  else e.addiu(R_FP, T0, static_cast<s32>(pre ? -static_cast<int>(n * 4) : -static_cast<int>(n * 4 - 4)));
  e.srl(R_FP, R_FP, 2); e.sll(R_FP, R_FP, 2);
  e.addiu(T1, R_FP, static_cast<s32>(n * 4 - 1));
  e.xor_(T1, T1, R_FP); e.srl(T1, T1, 11);
  slow.push_back(e.bnez(T1)); e.nop();
  e.srl(T1, R_FP, 11); e.sll(T1, T1, 2);
  e.load_ptr(T3, c.page_table.raw()); e.addu(T3, T3, T1);
  e.lw(T1, 0, T3); e.nop();
  e.sll(T2, T1, 2);
  slow.push_back(e.beq(T2, 0)); e.nop();
  e.srl(T3, T1, 30);
  if (load) e.andi(T3, T3, 1);
  slow.push_back(e.bnez(T3)); e.nop();
  e.addu(T2, T2, R_FP);

  if (writeback) {
    if (up) e.addiu(T3, T0, n * 4);
    else e.addiu(T3, T0, -static_cast<s32>(n * 4));
    e.sw(T3, 52, R_SP);
  }

  for (u32 r = 0, off = 0; r < 15; ++r) {
    if (!(list & (1u << r))) continue;
    if (load) {
      e.lw(T1, off, T2); e.nop(); putr(e, r, T1, cache_mask);
    } else {
      getr(e, T1, r, cache_mask); e.sw(T1, off, T2);
    }
    off += 4;
  }
  if (writeback) { e.lw(T3, 52, R_SP); e.nop(); putr(e, rn, T3, cache_mask); }
  if (c.which == ds::Cpu::ARM9) {
    // Aggregate N+S data timing; the ARM9 charges one CD/CDI.
    e.move(T0, 0);
    e.lw(T3, offsetof(CpuContext, timing9), R_CTX); e.nop();
    u32 k = 0;
    for (u32 r = 0; r < 15; ++r) {
      if (!(list & (1u << r))) continue;
      if (k) e.addiu(T1, R_FP, static_cast<s32>(k * 4));
      else e.move(T1, R_FP);
      e.srl(T1, T1, 12); e.sll(T1, T1, 3); e.addu(T1, T3, T1);
      e.lbu(T1, (load ? 0 : 4) + (k ? 3 : 2), T1); e.nop();
      e.addu(T0, T0, T1);
      ++k;
    }
    const u32 nc = arm9_code_cost(c, pc, false);
    if (nc >= 6) {
      e.sltiu(T1, T0, 6);
      const size_t short_data = e.bnez(T1); e.nop();
      e.addiu(T0, T0, static_cast<s32>(nc - 6));
      const size_t combined = e.b(); e.nop();
      e.patch_branch(short_data, e.size());
      li(e, T0, nc);
      e.patch_branch(combined, e.size());
    } else {
      if (nc) e.addiu(T1, 0, nc); else e.move(T1, 0);
      e.sltu(T3, T1, T0);
      e.subu(T3, 0, T3);
      e.xor_(T0, T0, T1); e.and_(T0, T0, T3); e.xor_(T0, T0, T1);
    }
    e.subu(R_BUDGET, R_BUDGET, T0);
  } else emit_arm7_block_cost(e, c, pc, R_FP, load, n);
  e.addiu(R_GUEST_PC, R_GUEST_PC, 4);
  ex.push_back(e.blez(R_BUDGET)); e.nop();
  done.push_back(e.b()); e.nop();

  const size_t slow_at = e.size();
  for (size_t p : slow) e.patch_branch(p, slow_at);
  e.move(4, R_CTX); li(e, 5, instr); li(e, 6, ds::jit::make_key(pc, false));
  e.load_ptr(25, reinterpret_cast<const void *>(&jit_h_fallback));
  spill_cached(e, cache_mask);
  e.sw(R_GUEST_PC, OR + 60, R_CTX); e.sw(R_BUDGET, OB, R_CTX);
  e.jalr(R_RA, 25); e.nop();
  e.lw(R_BUDGET, OB, R_CTX); e.nop();
  e.lw(R_GUEST_PC, OR + 60, R_CTX); e.nop();
  reload_cached(e, cache_mask);
  ex.push_back(e.b()); e.nop();

  const size_t not_exec_at = e.size();
  for (size_t p : not_executed) e.patch_branch(p, not_exec_at);
  emit_timing(e, c, pc, false);
  e.addiu(R_GUEST_PC, R_GUEST_PC, 4);
  ex.push_back(e.blez(R_BUDGET)); e.nop();
  const size_t end = e.size();
  for (size_t p : done) e.patch_branch(p, end);
}
static void emit_arm_msr_control(MipsEmitter &e, const CpuContext &c, u32 pc,
                                 u32 instr, u32 cache_mask,
                                 std::vector<size_t> &ex) {
  const u32 fields = (instr >> 16) & 15;
  const u32 mask = ((fields & 1) ? 0x000000dfu : 0) |
                   ((fields & 2) ? 0x0000ff00u : 0) |
                   ((fields & 4) ? 0x00ff0000u : 0) |
                   ((fields & 8) ? 0xff000000u : 0);
  std::vector<size_t> not_executed, changed_mode;
  emit_skip_if_false(e, instr >> 28, not_executed);
  e.lw(T0, ds::jit::OFF_CPSR, R_CTX); e.nop();
  if (ds::arm::decode_arm(instr) == AOp::MsrImm) li(e, T1, ai(instr));
  else getr(e, T1, instr & 15u, cache_mask);
  e.andi(T3, T0, 31); e.addiu(T2, 0, 0x10);
  const size_t user_mode = e.beq(T3, T2); e.nop();

  li(e, T2, mask); e.and_(T1, T1, T2);
  li(e, T2, ~mask); e.and_(T3, T0, T2); e.or_(T1, T1, T3);
  e.xor_(T2, T0, T1); e.andi(T2, T2, 31);
  changed_mode.push_back(e.bnez(T2)); e.nop();
  e.sw(T1, ds::jit::OFF_CPSR, R_CTX);
  const size_t user_done = e.b(); e.nop();

  const size_t user_at = e.size();
  e.patch_branch(user_mode, user_at);
  if (fields & 8) {
    li(e, T2, 0xff000000u); e.and_(T1, T1, T2);
    li(e, T2, 0x00ffffffu); e.and_(T0, T0, T2);
    e.or_(T1, T1, T0); e.sw(T1, ds::jit::OFF_CPSR, R_CTX);
  }
  e.patch_branch(user_done, e.size());
  emit_timing(e, c, pc, false);
  e.addiu(R_GUEST_PC, R_GUEST_PC, 4);
  ex.push_back(e.blez(R_BUDGET)); e.nop();
  ex.push_back(e.b()); e.nop(); // CPSR control changes may unmask an IRQ.

  const size_t slow_at = e.size();
  for (size_t p : changed_mode) e.patch_branch(p, slow_at);
  emit_timing(e, c, pc, false);
  e.addiu(R_GUEST_PC, R_GUEST_PC, 4);
  e.move(4, R_CTX); e.move(5, T1);
  e.load_ptr(25, reinterpret_cast<const void *>(&jit_h_mips_set_cpsr));
  spill_cached(e, cache_mask);
  e.sw(R_GUEST_PC, OR + 60, R_CTX); e.sw(R_BUDGET, OB, R_CTX);
  e.jalr(R_RA, 25); e.nop();
  e.lw(R_BUDGET, OB, R_CTX); e.nop();
  e.lw(R_GUEST_PC, OR + 60, R_CTX); e.nop();
  reload_cached(e, cache_mask);
  ex.push_back(e.b()); e.nop();

  const size_t not_exec_at = e.size();
  for (size_t p : not_executed) e.patch_branch(p, not_exec_at);
  emit_timing(e, c, pc, false);
  e.addiu(R_GUEST_PC, R_GUEST_PC, 4);
  ex.push_back(e.blez(R_BUDGET)); e.nop();
}
static void emit_timing(MipsEmitter &e, const CpuContext &c, u32 pc, bool thumb,
                        bool internal) {
  u32 cost;
  if (c.which == ds::Cpu::ARM9)
    cost = arm9_code_cost(c, pc, thumb);
  else {
    cost = c.timing7[pc >> 15][thumb ? (internal ? 0 : 1) : (internal ? 2 : 3)];
  }
  e.addiu(R_BUDGET, R_BUDGET, -static_cast<s32>(cost));
}
static void getr(MipsEmitter &e, u32 d, u32 r, u32 cache_mask) {
  if (r < 15 && (cache_mask & (1u << r))) {
    const u32 slot = __builtin_popcount(cache_mask & ((1u << r) - 1));
    e.move(d, CACHE_FIRST + slot);
    return;
  }
  e.lw(d, OR + 4 * r, R_CTX);
  e.nop();
}
static void putr(MipsEmitter &e, u32 r, u32 s, u32 cache_mask) {
  if (r < 15 && (cache_mask & (1u << r))) {
    const u32 slot = __builtin_popcount(cache_mask & ((1u << r) - 1));
    e.move(CACHE_FIRST + slot, s);
    return;
  }
  e.nop();
  e.sw(s, OR + 4 * r, R_CTX);
}
static void spill_cached(MipsEmitter &e, u32 cache_mask) {
  for (u32 r = 0; r < 15; ++r)
    if (cache_mask & (1u << r))
      e.sw(CACHE_FIRST + __builtin_popcount(cache_mask & ((1u << r) - 1)), OR + 4 * r, R_CTX);
}
static void reload_cached(MipsEmitter &e, u32 cache_mask) {
  for (u32 r = 0; r < 15; ++r) {
    if (!(cache_mask & (1u << r))) continue;
    e.lw(CACHE_FIRST + __builtin_popcount(cache_mask & ((1u << r) - 1)), OR + 4 * r, R_CTX);
    e.nop();
  }
}
static void note_code_dep(ds::jit::Block &b, u32 addr) {
  const u32 page = addr >> 12;
  for (u32 i = 0; i < b.ndep; ++i)
    if (b.dep_page[i] == page) {
      b.dep_kind[i] |= ds::mem::Timing::RETIME_CODE;
      return;
    }
  if (b.ndep < ds::jit::Block::DEP_MAX) {
    const u32 i = b.ndep;
    b.dep_page[i] = page;
    b.dep_kind[i] = ds::mem::Timing::RETIME_CODE;
    b.ndep = static_cast<ds::u8>(i + 1);
  } else
    b.dep_overflow = true;
}
static void note_data_dep(ds::jit::Block &b, u32 addr) {
  const u32 page = addr >> 12;
  for (u32 i = 0; i < b.ndep; ++i)
    if (b.dep_page[i] == page) {
      b.dep_kind[i] |= ds::mem::Timing::RETIME_DATA;
      return;
    }
  if (b.ndep < ds::jit::Block::DEP_MAX) {
    const u32 i = b.ndep;
    b.dep_page[i] = page;
    b.dep_kind[i] = ds::mem::Timing::RETIME_DATA;
    b.ndep = static_cast<ds::u8>(i + 1);
  } else
    b.dep_overflow = true;
}
static void emit_n(MipsEmitter &e, const N &n, u32 cache_mask) {
  if (n.op == 17) {
    getr(e, T0, n.rm, cache_mask);
    getr(e, T1, n.rn, cache_mask);
    if (n.signed_mul) e.mult(T0, T1); else e.multu(T0, T1);
    e.nop(); e.nop(); e.mflo(T2); e.mfhi(T3);
    if (n.accumulate) {
      getr(e, T0, n.rdlo, cache_mask);
      getr(e, T1, n.rd, cache_mask);
      e.addu(T2, T2, T0);
      e.sltu(T0, T2, T0);
      e.addu(T3, T3, T1);
      e.addu(T3, T3, T0);
    }
    if (n.setflags) {
      e.lw(T0, ds::jit::OFF_CPSR, R_CTX); e.nop();
      li(e, T1, 0x3fffffffu); e.and_(T0, T0, T1);
      e.srl(T1, T3, 31); e.sll(T1, T1, 31); e.or_(T0, T0, T1);
      e.or_(T1, T3, T2);
      e.sltiu(T1, T1, 1); e.sll(T1, T1, 30); e.or_(T0, T0, T1);
      e.sw(T0, ds::jit::OFF_CPSR, R_CTX);
    }
    putr(e, n.rdlo, T2, cache_mask);
    putr(e, n.rd, T3, cache_mask);
    return;
  }
  if (n.op == 16) {
    getr(e, T0, n.rm, cache_mask);
    getr(e, T1, n.rn, cache_mask);
    e.multu(T0, T1); e.nop(); e.nop(); e.mflo(T2);
    if (n.accumulate) { getr(e, T0, n.rdlo, cache_mask); e.addu(T2, T2, T0); }
    if (n.setflags) {
      e.lw(T0, ds::jit::OFF_CPSR, R_CTX); e.nop();
      li(e, T1, n.mul_arm7 ? 0x1fffffffu : 0x3fffffffu); e.and_(T0, T0, T1);
      e.srl(T1, T2, 31); e.sll(T1, T1, 31); e.or_(T0, T0, T1);
      e.sltiu(T1, T2, 1); e.sll(T1, T1, 30); e.or_(T0, T0, T1);
      e.sw(T0, ds::jit::OFF_CPSR, R_CTX);
    }
    putr(e, n.rd, T2, cache_mask);
    return;
  }
  if (n.op != 13) {
    if (n.rn_pc) e.move(T0, R_GUEST_PC);
    else getr(e, T0, n.rn, cache_mask);
  }
  if (n.im)
    li(e, T1, n.imm);
  else {
    if (n.rm_pc) e.move(T1, R_GUEST_PC);
    else getr(e, T1, n.rm, cache_mask);
    if (n.reg_shift) {
      if (n.carry_dynamic) e.move(12, T1); // keep the unshifted value for carry-out
      getr(e, T3, n.shift_reg, cache_mask);
      e.andi(T3, T3, 255);
      if (n.shift_type == 0 || n.shift_type == 1) {
        if (n.shift_type == 0) e.sllv(T1, T1, T3);
        else e.srlv(T1, T1, T3);
        const size_t in_range = e.beq(T3, 0); e.nop();
        // A zero low-five-bit count is either zero or a multiple of 32;
        // only the latter saturates LSL/LSR to zero.
        li(e, T2, 32);
        e.sltu(T2, T3, T2);
        const size_t small = e.bnez(T2); e.nop();
        e.move(T1, 0);
        const size_t done = e.b(); e.nop();
        e.patch_branch(in_range, e.size());
        // Counts 1..31 already produced their result.  For count 32, force 0.
        e.patch_branch(small, e.size());
        e.patch_branch(done, e.size());
      } else if (n.shift_type == 2) {
        e.srav(T1, T1, T3);
        li(e, T2, 32);
        e.sltu(T2, T3, T2);
        const size_t small = e.bnez(T2); e.nop();
        e.sra(T1, T1, 31);
        e.patch_branch(small, e.size());
      } else {
        e.rotrv(T1, T1, T3);
      }
      if (n.carry_dynamic) {
        e.move(14, T3);
        e.lw(T3, ds::jit::OFF_CPSR, R_CTX); e.nop();
        e.srl(T3, T3, 29); e.andi(T3, T3, 1);
        const size_t no_shift = e.beq(14, 0); e.nop();
        std::vector<size_t> carry_done;
        if (n.shift_type == 0 || n.shift_type == 1) {
          li(e, 13, 32);
          if (n.shift_type == 0) e.subu(13, 13, 14);
          else e.addiu(13, 14, -1);
          if (n.shift_type == 0) {
            li(e, 13, 32); e.sltu(13, 13, 14);
            const size_t within = e.beq(13, 0); e.nop();
            e.move(T3, 0);
            carry_done.push_back(e.b()); e.nop();
            e.patch_branch(within, e.size());
            li(e, 13, 32); e.subu(13, 13, 14);
            e.srlv(T3, 12, 13); e.andi(T3, T3, 1);
          } else {
            li(e, 13, 32); e.sltu(13, 13, 14);
            const size_t within = e.beq(13, 0); e.nop();
            e.move(T3, 0);
            carry_done.push_back(e.b()); e.nop();
            e.patch_branch(within, e.size());
            e.addiu(13, 14, -1); e.srlv(T3, 12, 13); e.andi(T3, T3, 1);
          }
        } else if (n.shift_type == 2) {
          li(e, 13, 32); e.sltu(13, 14, 13);
          const size_t small = e.bnez(13); e.nop();
          e.srl(T3, 12, 31); e.andi(T3, T3, 1);
          carry_done.push_back(e.b()); e.nop();
          e.patch_branch(small, e.size());
          e.addiu(13, 14, -1); e.srlv(T3, 12, 13); e.andi(T3, T3, 1);
        } else {
          e.andi(13, 14, 31);
          const size_t rotate_zero = e.beq(13, 0); e.nop();
          e.addiu(13, 13, -1); e.srlv(T3, 12, 13); e.andi(T3, T3, 1);
          carry_done.push_back(e.b()); e.nop();
          e.patch_branch(rotate_zero, e.size());
          e.srl(T3, 12, 31); e.andi(T3, T3, 1);
        }
        for (size_t p : carry_done) e.patch_branch(p, e.size());
        e.patch_branch(no_shift, e.size());
      }
    }
    if (n.carry_dynamic && !n.reg_shift) {
      const u32 bit = n.shift_type == 0 ? 32 - n.shift :
                      n.shift_type == 3 && !n.shift ? 0 :
                      n.shift ? n.shift - 1 : 31;
      e.srl(T3, T1, bit);
      e.andi(T3, T3, 1);
    }
    if (!n.reg_shift) {
      if (n.shift) {
        if (n.shift_type == 1)
          e.srl(T1, T1, n.shift);
        else if (n.shift_type == 2)
          e.sra(T1, T1, n.shift);
        else if (n.shift_type == 3)
          e.rotr(T1, T1, n.shift);
        else
          e.sll(T1, T1, n.shift);
      } else if (n.shift_type == 1) {
        e.addiu(T1, 0, 0); // LSR #32
      } else if (n.shift_type == 2) {
        e.sra(T1, T1, 31); // ASR #32
      } else if (n.shift_type == 3) {
        e.lw(T2, ds::jit::OFF_CPSR, R_CTX); e.nop();
        e.srl(T2, T2, 29); e.andi(T2, T2, 1); e.sll(T2, T2, 31);
        e.srl(T1, T1, 1); e.or_(T1, T1, T2);
      }
    }
  }
  switch (n.op) {
  case 0:
    e.and_(T2, T0, T1);
    break;
  case 1:
    e.xor_(T2, T0, T1);
    break;
  case 2:
  case 10:
    e.subu(T2, T0, T1);
    break;
  case 3:
    e.subu(T2, T1, T0);
    break;
  case 4:
  case 11:
    // Some 74Kc/XBurst revisions mis-execute back-to-back generated ADDU.
    // Equivalent modulo-32-bit form keeps ADD native without that opcode.
    if (n.setflags) {
      e.nor(T3, T1, 0);
      e.subu(T2, T0, T3);
      e.addiu(T2, T2, -1);
    } else {
      e.nor(T1, T1, 0);
      e.subu(T2, T0, T1);
      e.addiu(T2, T2, -1);
    }
    break;
  case 5: {
    e.move(12, T1);
    e.lw(T3, ds::jit::OFF_CPSR, R_CTX);
    e.nop();
    e.srl(T3, T3, 29);
    e.andi(T3, T3, 1);
    e.nor(T1, T1, 0);
    e.subu(T2, T0, T1);
    e.addiu(T2, T2, -1);
    e.move(13, T2);
    e.subu(T1, 0, T3);
    e.subu(T2, T2, T1);
    e.sltu(T1, T2, 13);
    e.sltu(T3, 13, T0);
    e.or_(T3, T3, T1);
    e.sll(T3, T3, 29);
    break;
  }
  case 6: {
    e.move(12, T1);
    e.lw(T3, ds::jit::OFF_CPSR, R_CTX);
    e.nop();
    e.srl(T3, T3, 29);
    e.andi(T3, T3, 1);
    e.xori(T3, T3, 1);
    e.move(13, T3);
    e.subu(T2, T0, T1);
    e.move(T1, T2);
    e.sltu(T3, T0, 12);
    e.subu(T2, T2, 13);
    e.sltu(T1, T1, 13);
    e.or_(T3, T3, T1);
    e.xori(T3, T3, 1);
    e.sll(T3, T3, 29);
    break;
  }
  case 7: {
    e.move(12, T1);
    e.lw(T3, ds::jit::OFF_CPSR, R_CTX);
    e.nop();
    e.srl(T3, T3, 29);
    e.andi(T3, T3, 1);
    e.xori(T3, T3, 1);
    e.move(13, T3);
    e.subu(T2, T1, T0);
    e.move(T1, T2);
    e.sltu(T3, 12, T0);
    e.subu(T2, T2, 13);
    e.sltu(T1, T1, 13);
    e.or_(T3, T3, T1);
    e.xori(T3, T3, 1);
    e.sll(T3, T3, 29);
    break;
  }
  case 8:
    e.and_(T2, T0, T1);
    break;
  case 9:
    e.xor_(T2, T0, T1);
    break;
  case 12:
    e.or_(T2, T0, T1);
    break;
  case 13:
    e.move(T2, T1);
    break;
  case 14:
    e.subu(12, 0, T1); e.addiu(12, 12, -1);
    e.and_(T2, T0, 12);
    break;
  case 15:
    e.subu(T2, 0, T1); e.addiu(T2, T2, -1);
    break;
  }
  if (n.setflags) {
    const bool arithmetic = n.op == 2 || n.op == 3 || n.op == 4 || n.op == 5 ||
                            n.op == 6 || n.op == 7 || n.op == 10 || n.op == 11;
    if (arithmetic) {
      if (n.op == 4 || n.op == 5 || n.op == 11) {
        if (n.op == 5) {
          // C was calculated together with the two-stage addition above.
        } else {
        e.sltu(T3, T2, T0);
        e.andi(T3, T3, 1);
        e.sll(T3, T3, 29);
        }
      } else {
        if (n.op == 2 || n.op == 10) {
          e.sltu(T3, T0, T1);
          e.nor(T3, T3, 0);
          e.andi(T3, T3, 1);
          e.sll(T3, T3, 29);
        } else if (n.op == 3) {
          e.sltu(T3, T1, T0);
          e.nor(T3, T3, 0);
          e.andi(T3, T3, 1);
          e.sll(T3, T3, 29);
        }
      }
      const bool reverse = n.op == 3 || n.op == 7;
      const u32 lhs = reverse ? (n.op == 3 ? T1 : 12) : T0;
      const u32 rhs = n.op == 5 || n.op == 6 || n.op == 7 ?
                      (n.op == 7 ? T0 : 12) : (reverse ? T0 : T1);
      e.xor_(13, lhs, rhs);
      if (n.op == 4 || n.op == 5 || n.op == 11) e.nor(13, 13, 0);
      e.xor_(12, lhs, T2);
      e.and_(12, 13, 12);
      e.srl(12, 12, 31);
      e.sll(12, 12, 28);
      e.or_(T3, T3, 12);
      e.lw(T1, ds::jit::OFF_CPSR, R_CTX);
      e.nop();
      li(e, T0, 0x0fffffffu);
      e.and_(T1, T1, T0);
      e.srl(T0, T2, 31);
      e.sll(T0, T0, 31);
      e.or_(T1, T1, T0);
      e.sltiu(T0, T2, 1);
      e.sll(T0, T0, 30);
      e.or_(T1, T1, T0);
      e.or_(T1, T1, T3);
      e.sw(T1, ds::jit::OFF_CPSR, R_CTX);
    } else {
    e.lw(T1, ds::jit::OFF_CPSR, R_CTX);
    li(e, T0, n.carry_valid ? 0x1fffffffu : 0x3fffffffu);
    e.and_(T1, T1, T0);
    e.srl(T0, T2, 31);
    e.sll(T0, T0, 31);
    e.or_(T1, T1, T0);
    e.sltiu(T0, T2, 1);
    e.sll(T0, T0, 30);
    e.or_(T1, T1, T0);
    if (n.carry_dynamic) {
      const size_t no_carry = e.beq(T3, 0);
      e.nop();
      li(e, T0, 0x20000000u);
      e.or_(T1, T1, T0);
      e.patch_branch(no_carry, e.size());
    } else if (n.carry_valid && n.carry) {
      li(e, T0, 0x20000000u);
      e.or_(T1, T1, T0);
    }
    e.sw(T1, ds::jit::OFF_CPSR, R_CTX);
    }
  }
  if (n.op != 8 && n.op != 9 && n.op != 10 && n.op != 11)
    putr(e, n.rd, T2, cache_mask);
}
static bool local_loop_branch(u32 x, u32 pc, u32 start) {
  if ((x & 0x0e000000u) != 0x0a000000u || (x & (1u << 24)))
    return false;
  const s32 off = static_cast<s32>(x << 8) >> 6;
  return pc + 8 + static_cast<u32>(off) == start;
}
static bool arm_branch(u32 x) {
  return (x >> 28) != 15 && (x & 0x0e000000u) == 0x0a000000u;
}
static bool native_ignored_mcr(u32 x, bool arm9) {
  if (!arm9 || ds::arm::decode_arm(x) != AOp::Mcr || ((x >> 8) & 15) != 15)
    return false;
  return ds::jit::shape::mcr_is_nop(x, true) || ((x >> 16) & 15) == 0;
}
static bool native_mrs_cpsr(u32 x) {
  return ds::arm::decode_arm(x) == AOp::Mrs && !(x & (1u << 22)) &&
         ((x >> 12) & 15u) != 15u && (x >> 28) != 15u;
}
static bool native_mrs_spsr(u32 x) {
  return ds::arm::decode_arm(x) == AOp::Mrs && (x & (1u << 22)) &&
         ((x >> 12) & 15u) != 15u && (x >> 28) != 15u;
}
static bool native_msr_flags(u32 x) {
  const AOp op = ds::arm::decode_arm(x);
  return (op == AOp::MsrReg || op == AOp::MsrImm) &&
         !(x & (1u << 22)) && ((x >> 16) & 15u) == 8 && (x >> 28) != 15 &&
         (op != AOp::MsrReg || (x & 15u) != 15u);
}
static bool native_msr_control(u32 x) {
  const AOp op = ds::arm::decode_arm(x);
  return (op == AOp::MsrReg || op == AOp::MsrImm) && !(x & (1u << 22)) &&
         ((x >> 16) & 1u) && (x >> 28) != 15u &&
         (op != AOp::MsrReg || (x & 15u) != 15u);
}
static bool native_msr_spsr(u32 x) {
  const AOp op = ds::arm::decode_arm(x);
  return (op == AOp::MsrReg || op == AOp::MsrImm) && (x & (1u << 22)) &&
         ((x >> 16) & 15u) && (x >> 28) != 15u &&
         (op != AOp::MsrReg || (x & 15u) != 15u);
}
static bool native_arm_exc_return_reg(u32 x) {
  return ds::arm::decode_arm(x) == AOp::DpImmShift && (x >> 28) == 14 &&
         (x & (1u << 20)) && ((x >> 21) & 15u) == 13 &&
         ((x >> 12) & 15u) == 15 && !(x & 0xff0u) && (x & 15u) != 15u;
}
static bool native_arm_exc_return_imm(u32 x) {
  if (ds::arm::decode_arm(x) != AOp::DpImm || (x >> 28) != 14 ||
      !(x & (1u << 20)) || ((x >> 12) & 15u) != 15u)
    return false;
  const u32 op = (x >> 21) & 15u;
  return op == 2 || op == 4;
}
static bool native_arm_exc_return(u32 x) {
  return native_arm_exc_return_reg(x) || native_arm_exc_return_imm(x);
}
static bool thumb_branch(u32 x) {
  return ((x & 0xf000u) == 0xd000u && ((x >> 8) & 15u) < 14u) ||
         (x & 0xf800u) == 0xe000u;
}
static bool thumb_bl_prefix(u32 x) { return (x & 0xf800u) == 0xf000u; }
static bool thumb_bl_suffix(u32 x) { return (x & 0xf800u) == 0xf800u; }
static bool native_thumb_special(u32 x, u32 pc, N &n) {
  const u32 h = static_cast<ds::u16>(x);
  if ((h & 0xf000u) == 0xa000u) {
    n.op = (h & 0x0800u) ? 4 : 13;
    n.rd = (h >> 8) & 7;
    n.rn = 13;
    n.im = true;
    n.setflags = false;
    n.imm = (h & 255) << 2;
    if (!(h & 0x0800u)) n.imm += (pc + 4) & ~3u;
    return true;
  }
  if ((h & 0xff00u) == 0xb000u) {
    n.op = (h & 0x80u) ? 2 : 4;
    n.rd = n.rn = 13;
    n.im = true;
    n.setflags = false;
    n.imm = (h & 0x7f) << 2;
    return true;
  }
  return false;
}
struct B {
  size_t ip, kp, br;
  std::vector<u32> i, k;
};
} // namespace

extern "C" ds::u32 ds_jit_mips_fallback_block(ds::CpuContext *c,
                                              const ds::u32 *ins,
                                              const ds::u32 *keys, ds::u32 n) {
  for (ds::u32 i = 0; i < n; ++i) {
    if (ds::jit::jit_h_fallback(c, ins[i], keys[i]) ||
        c->hot.cycle_budget <= 0 || c->halted || c->hot.alerts)
      return 1;
  }
  return 0;
}
namespace ds::jit::backend {
bool translate_block(JitCpu &jc, u32 key, u8 *buf, size_t cap, Block &b,
                     u32 &size) {
  const u32 start = key_pc(key);
  const bool thumb = key_thumb(key);
  if (!jc.ctx->page_table.read_ptr(start) || cap < 128)
    return false;
  const bool arm9 = jc.ctx->which == ds::Cpu::ARM9;
  const char *native_env = std::getenv("DS_MIPS_NATIVE");
  const char *native_arm7_env = std::getenv("DS_MIPS_NATIVE_ARM7");
  const bool use_native = native_env && std::strcmp(native_env, "0") &&
                          (arm9 || (native_arm7_env && std::strcmp(native_arm7_env, "0")));
  const char *branch_env = std::getenv("DS_MIPS_NATIVE_BRANCH");
  const bool use_native_branch = use_native && branch_env && std::strcmp(branch_env, "0");
  const char *mem_env = std::getenv("DS_MIPS_NATIVE_MEM");
  const char *mem_arm7_env = std::getenv("DS_MIPS_NATIVE_MEM_ARM7");
  const bool use_native_mem = use_native && mem_env && std::strcmp(mem_env, "0") &&
                              (arm9 || !mem_arm7_env || std::strcmp(mem_arm7_env, "0"));
  const char *limit_env = std::getenv("DS_MIPS_NATIVE_LIMIT");
  const u32 native_limit = limit_env ? std::strtoul(limit_env, nullptr, 10) : 0;
  const bool native_trace = std::getenv("DS_MIPS_NATIVE_TRACE") &&
                            std::strcmp(std::getenv("DS_MIPS_NATIVE_TRACE"), "0");
  u32 native_count = 0;
  std::vector<u32> is, ks;
  u32 addr = start;
  const u32 page = start & ~0xfffu;
  for (u32 i = 0; i < 64 && ((addr & ~0xfffu) == page); i++) {
    u8 *p = jc.ctx->page_table.read_ptr(addr);
    if (!p)
      break;
    u32 x = 0;
    if (thumb)
      std::memcpy(&x, p, 2);
    else
      std::memcpy(&x, p, 4);
    const bool br = !thumb &&
        ((x & 0x0e000000u) == 0x0a000000u || (x & 0x0ffffff0u) == 0x012fff10u);
    is.push_back(x);
    ks.push_back(make_key(addr, thumb));
    addr += thumb ? 2 : 4;
    if (br)
      break;
  }
  if (is.empty())
    return false;
  u32 candidates = 0, cache_mask = 0;
  u32 reg_score[15] = {};
  auto score_reg = [&](u32 r, u32 weight = 1) {
    if (r < 15) reg_score[r] += weight;
  };
  for (u32 i = 0; i < is.size(); ++i) {
    const u32 x = is[i];
    N n{};
    if (use_native && (!native_limit || candidates < native_limit) &&
        native_insn(x, thumb, arm9, n)) {
      ++candidates;
      if (n.op != 13 || n.reg_shift) score_reg(n.rn);
      if (!n.im) score_reg(n.rm);
      if (n.reg_shift) score_reg(n.shift_reg);
      if (n.op == 17 && n.accumulate) score_reg(n.rd);
      if (n.op == 17 || (n.op == 16 && n.accumulate)) score_reg(n.rdlo);
    }
    if (use_native && !thumb &&
        (!native_limit || candidates < native_limit) &&
        native_ignored_mcr(x, arm9))
      ++candidates;
    if (use_native && !thumb &&
        (!native_limit || candidates < native_limit) && native_mrs_cpsr(x)) {
      ++candidates;
    }
    if (use_native && !thumb &&
        (!native_limit || candidates < native_limit) && native_msr_flags(x)) {
      ++candidates;
      if (ds::arm::decode_arm(x) == AOp::MsrReg) score_reg(x & 15u);
    }
    ThumbMem m{};
    bool mem_candidate = thumb ? native_thumb_mem(x, m) :
                                native_arm_mem(x, key_pc(ks[i]), m, arm9);
    if (!mem_candidate && !thumb)
      mem_candidate = native_arm_single_multi(x, m);
    if (mem_candidate && !arm9)
      mem_candidate = arm7_data_slot(*jc.ctx, key_pc(ks[i]), thumb,
                                     m.width, m.load) >= 0;
    if (use_native_mem && (!native_limit || candidates < native_limit) && mem_candidate) {
      ++candidates;
      score_reg(m.rn);
      if (m.reg) score_reg(m.rm);
      if (!m.load) score_reg(m.rd);
    }
    if (use_native_branch && !thumb &&
        (!native_limit || candidates < native_limit) && native_arm_exc_return(x)) {
      ++candidates;
      if (native_arm_exc_return_reg(x)) score_reg(x & 15u);
      else if (((x >> 21) & 15u) != 13 && ((x >> 16) & 15u) != 15u)
        score_reg((x >> 16) & 15u);
    }
  }
  if (use_native_branch && !thumb &&
      (!native_limit || candidates < native_limit) &&
      (is.back() & 0x0FFFFFF0u) == 0x012FFF10u)
    score_reg(is.back() & 15u);
  // Keep the six guest registers that eliminate the most context loads/stores.
  for (u32 slot = 0; slot < CACHE_COUNT; ++slot) {
    u32 best = 15;
    for (u32 r = 0; r < 15; ++r)
      if (!(cache_mask & (1u << r)) &&
          (best == 15 || reg_score[r] > reg_score[best])) best = r;
    if (best == 15 || reg_score[best] < 5) break;
    cache_mask |= 1u << best;
  }
  const bool cache_regs = cache_mask != 0;
  const bool has_native = use_native;
  MipsEmitter e(buf, cap);
  e.addiu(R_SP, R_SP, -static_cast<s32>(has_native ? CACHE_FRAME : 24));
  e.sw(R_CTX, 16, R_SP);
  e.sw(R_RA, 20, R_SP);
  if (has_native) {
    e.sw(R_BUDGET, 48, R_SP);
    e.sw(R_FP, 56, R_SP);
  }
  if (cache_regs) {
    for (u32 r = 0; r < 15; ++r)
      if (cache_mask & (1u << r))
        e.sw(CACHE_FIRST + __builtin_popcount(cache_mask & ((1u << r) - 1)),
             CACHE_SAVED + 4 * __builtin_popcount(cache_mask & ((1u << r) - 1)), R_SP);
  }
  e.move(R_CTX, 4);
  if (has_native) {
    e.lw(R_BUDGET, OB, R_CTX);
    e.nop();
    li(e, R_GUEST_PC, start + (thumb ? 4 : 8));
  }
  if (cache_regs)
    reload_cached(e, cache_mask);
  std::vector<B> bs;
  std::vector<size_t> ex;
  std::vector<size_t> bx;
  auto emit_link = [&](u32 target, bool target_thumb) {
    spill_cached(e, cache_mask);
    e.sw(R_BUDGET, OB, R_CTX);
    e.sw(R_GUEST_PC, OR + 60, R_CTX);
    e.move(4, R_CTX);
    e.lw(T3, 20, R_SP); e.nop();
    e.sw(T3, static_cast<s32>(CACHE_FRAME + 12), R_SP);
    e.move(R_RA, T3);
    if (cache_regs) {
      for (u32 r = 0; r < 15; ++r) {
        if (!(cache_mask & (1u << r))) continue;
        const u32 slot = __builtin_popcount(cache_mask & ((1u << r) - 1));
        e.lw(CACHE_FIRST + slot, CACHE_SAVED + 4 * slot, R_SP); e.nop();
      }
    }
    e.lw(R_BUDGET, 48, R_SP); e.nop();
    e.lw(R_FP, 56, R_SP); e.nop();
    e.lw(R_CTX, 16, R_SP); e.nop();
    e.addiu(R_SP, R_SP, CACHE_FRAME);
    e.jal(reinterpret_cast<uintptr_t>(jc.link)); e.nop();
    e.w(ds::jit::make_key(target, target_thumb));
  };
  const size_t loop_head = e.size();
  for (u32 q = 0; q < is.size();) {
    const u32 pc = key_pc(ks[q]);
    if (use_native_branch && thumb && q + 1 < is.size() &&
        (!native_limit || native_count + 2 <= native_limit) &&
        thumb_bl_prefix(is[q]) && thumb_bl_suffix(is[q + 1])) {
      note_code_dep(b, pc + 4);
      note_code_dep(b, pc + 6);
      emit_timing(e, *jc.ctx, pc, true);
      const s32 high = static_cast<s32>((is[q] & 0x7ffu) << 21) >> 9;
      const u32 target = pc + 4 + static_cast<u32>(high) +
                         ((is[q + 1] & 0x7ffu) << 1);
      li(e, T0, pc + 5);
      e.sw(T0, OR + 56, R_CTX);
      li(e, T0, target + 4);
      e.move(R_GUEST_PC, T0);
      u32 code_after = 0;
      const u32 refill = ds::refill_cycles(*jc.ctx, target, true, &code_after);
      if (refill) e.addiu(R_BUDGET, R_BUDGET, -static_cast<s32>(refill));
      if (arm9) {
        li(e, T0, code_after);
        e.sw(T0, offsetof(CpuContext, code_cycles), R_CTX);
        e.addiu(T0, 0, 0);
        e.sb(T0, offsetof(CpuContext, branch_fetch), R_CTX);
      } else {
        li(e, T0, target >> 15);
        e.sw(T0, offsetof(CpuContext, code_cycles), R_CTX);
        li(e, T0, target >> 24);
        e.sw(T0, offsetof(CpuContext, code_region), R_CTX);
      }
      ex.push_back(e.blez(R_BUDGET));
      e.nop();
      emit_link(target, true);
      native_count += 2;
      rt().stats.instrs_translated += 2;
      q += 2;
      continue;
    }
    if (use_native_branch && !thumb && (!native_limit || native_count < native_limit) &&
        local_loop_branch(is[q], pc, start)) {
      note_code_dep(b, pc + 8);
      std::vector<size_t> not_taken;
      emit_skip_if_false(e, is[q] >> 28, not_taken);
      li(e, T0, start + 8);
      e.move(R_GUEST_PC, T0);
      const u32 target = start;
      u32 code_after = 0;
      const u32 refill = ds::refill_cycles(*jc.ctx, target, false, &code_after);
      if (refill) {
        e.addiu(R_BUDGET, R_BUDGET, -static_cast<s32>(refill));
      }
      if (arm9) {
        li(e, T0, code_after);
        e.sw(T0, offsetof(CpuContext, code_cycles), R_CTX);
        e.addiu(T0, 0, 0);
        e.sb(T0, offsetof(CpuContext, branch_fetch), R_CTX);
      } else {
        li(e, T0, target >> 15);
        e.sw(T0, offsetof(CpuContext, code_cycles), R_CTX);
        li(e, T0, target >> 24);
        e.sw(T0, offsetof(CpuContext, code_region), R_CTX);
      }
      ex.push_back(e.blez(R_BUDGET));
      e.nop();
      const size_t loop = e.b();
      e.nop();
      e.patch_branch(loop, loop_head);
      for (size_t p : not_taken) e.patch_branch(p, e.size());
      li(e, T0, pc + 12);
      e.move(R_GUEST_PC, T0);
      emit_timing(e, *jc.ctx, pc, false);
      ex.push_back(e.blez(R_BUDGET));
      e.nop();
      ++native_count;
      rt().stats.instrs_translated++;
      ++q;
      continue;
    }
    if (use_native_branch && !thumb &&
        (!native_limit || native_count < native_limit) && arm_branch(is[q])) {
      note_code_dep(b, pc + 8);
      std::vector<size_t> not_taken;
      emit_skip_if_false(e, is[q] >> 28, not_taken);
      if (is[q] & (1u << 24)) {
        li(e, T0, pc + 4);
        e.sw(T0, OR + 56, R_CTX);
      }
      const s32 off = static_cast<s32>(is[q] << 8) >> 6;
      const u32 target = pc + 8 + static_cast<u32>(off);
      li(e, T0, target + 8);
      e.move(R_GUEST_PC, T0);
      u32 code_after = 0;
      const u32 refill = ds::refill_cycles(*jc.ctx, target, false, &code_after);
      if (refill) e.addiu(R_BUDGET, R_BUDGET, -static_cast<s32>(refill));
      if (arm9) {
        li(e, T0, code_after);
        e.sw(T0, offsetof(CpuContext, code_cycles), R_CTX);
        e.addiu(T0, 0, 0);
        e.sb(T0, offsetof(CpuContext, branch_fetch), R_CTX);
      } else {
        li(e, T0, target >> 15);
        e.sw(T0, offsetof(CpuContext, code_cycles), R_CTX);
        li(e, T0, target >> 24);
        e.sw(T0, offsetof(CpuContext, code_region), R_CTX);
      }
      ex.push_back(e.blez(R_BUDGET));
      e.nop();
      emit_link(target, false);
      for (size_t p : not_taken) e.patch_branch(p, e.size());
      emit_timing(e, *jc.ctx, pc, false);
      li(e, T0, pc + 12);
      e.move(R_GUEST_PC, T0);
      ex.push_back(e.blez(R_BUDGET));
      e.nop();
      ++native_count;
      rt().stats.instrs_translated++;
      ++q;
      continue;
    }
    if (use_native_branch && !thumb &&
        (!native_limit || native_count < native_limit) &&
        native_arm_exc_return(is[q])) {
      note_code_dep(b, pc + 8);
      emit_timing(e, *jc.ctx, pc, false);
      if (native_arm_exc_return_reg(is[q])) {
        getr(e, T0, is[q] & 15u, cache_mask);
      } else {
        const u32 op = (is[q] >> 21) & 15u;
        const u32 rn = (is[q] >> 16) & 15u;
        if (rn == 15) e.move(T0, R_GUEST_PC);
        else getr(e, T0, rn, cache_mask);
        li(e, T1, ai(is[q]));
        if (op == 2) e.subu(T0, T0, T1); else e.addu(T0, T0, T1);
      }
      e.move(4, R_CTX); e.move(5, T0);
      e.load_ptr(25, reinterpret_cast<const void *>(&jit_h_mips_exc_return));
      spill_cached(e, cache_mask);
      e.sw(R_GUEST_PC, OR + 60, R_CTX); e.sw(R_BUDGET, OB, R_CTX);
      e.jalr(R_RA, 25); e.nop();
      e.move(R_GUEST_PC, 2);
      e.lw(R_BUDGET, OB, R_CTX); e.nop();
      reload_cached(e, cache_mask);
      bx.push_back(e.b()); e.nop();
      ++native_count;
      rt().stats.instrs_translated++;
      ++q;
      continue;
    }
    if (use_native_branch && !thumb &&
        (!native_limit || native_count < native_limit) &&
        (is[q] & 0x0FFFFFF0u) == 0x012FFF10u && (is[q] & 15u) != 15u) {
      std::vector<size_t> not_taken;
      emit_skip_if_false(e, is[q] >> 28, not_taken);
      getr(e, T0, is[q] & 15u, cache_mask);
      e.move(4, R_CTX);
      e.move(5, T0);
      e.addiu(6, 0, 1);
      e.load_ptr(25, reinterpret_cast<const void *>(&jit_h_mips_indirect_branch));
      e.sw(R_BUDGET, OB, R_CTX);
      e.jalr(R_RA, 25); e.nop();
      e.move(R_GUEST_PC, 2);
      e.lw(R_BUDGET, OB, R_CTX); e.nop();
      const size_t taken = e.b(); e.nop();
      for (size_t p : not_taken) e.patch_branch(p, e.size());
      emit_timing(e, *jc.ctx, pc, false);
      e.addiu(R_GUEST_PC, R_GUEST_PC, 4);
      ex.push_back(e.blez(R_BUDGET)); e.nop();
      bx.push_back(taken);
      ++native_count;
      rt().stats.instrs_translated++;
      ++q;
      continue;
    }
    if (use_native_branch && thumb &&
        (!native_limit || native_count < native_limit) && thumb_branch(is[q])) {
      note_code_dep(b, pc + 4);
      const bool conditional = (is[q] & 0xf000u) == 0xd000u;
      std::vector<size_t> not_taken;
      if (conditional) emit_skip_if_false(e, (is[q] >> 8) & 15u, not_taken);
      const s32 off = conditional ? static_cast<s8>(is[q]) * 2 :
                                    (static_cast<s32>(is[q] << 21) >> 20);
      const u32 target = pc + 4 + static_cast<u32>(off);
      li(e, T0, target + 4);
      e.move(R_GUEST_PC, T0);
      u32 code_after = 0;
      const u32 refill = ds::refill_cycles(*jc.ctx, target, true, &code_after);
      if (refill) e.addiu(R_BUDGET, R_BUDGET, -static_cast<s32>(refill));
      if (arm9) {
        li(e, T0, code_after);
        e.sw(T0, offsetof(CpuContext, code_cycles), R_CTX);
        e.addiu(T0, 0, 0);
        e.sb(T0, offsetof(CpuContext, branch_fetch), R_CTX);
      } else {
        li(e, T0, target >> 15);
        e.sw(T0, offsetof(CpuContext, code_cycles), R_CTX);
        li(e, T0, target >> 24);
        e.sw(T0, offsetof(CpuContext, code_region), R_CTX);
      }
      ex.push_back(e.blez(R_BUDGET));
      e.nop();
      emit_link(target, true);
      for (size_t p : not_taken) e.patch_branch(p, e.size());
      if (conditional) {
        emit_timing(e, *jc.ctx, pc, true);
        li(e, T0, pc + 6);
        e.move(R_GUEST_PC, T0);
        ex.push_back(e.blez(R_BUDGET));
        e.nop();
      }
      ++native_count;
      rt().stats.instrs_translated++;
      ++q;
      continue;
    }
    if (use_native && !thumb &&
        (!native_limit || native_count < native_limit) &&
        native_ignored_mcr(is[q], arm9)) {
      note_code_dep(b, pc + 8);
      emit_timing(e, *jc.ctx, pc, false);
      std::vector<size_t> skip;
      emit_skip_if_false(e, is[q] >> 28, skip);
      e.addiu(R_BUDGET, R_BUDGET, -2); // CP15 MCR uses charge_CI(2).
      for (size_t p : skip) e.patch_branch(p, e.size());
      e.addiu(R_GUEST_PC, R_GUEST_PC, 4);
      ex.push_back(e.blez(R_BUDGET)); e.nop();
      ++native_count;
      rt().stats.instrs_translated++;
      ++q;
      continue;
    }
    if (use_native && !thumb &&
        (!native_limit || native_count < native_limit) &&
        (native_mrs_cpsr(is[q]) || native_mrs_spsr(is[q]))) {
      note_code_dep(b, pc + 8);
      emit_timing(e, *jc.ctx, pc, false);
      std::vector<size_t> skip;
      emit_skip_if_false(e, is[q] >> 28, skip);
      const u32 status = (is[q] & (1u << 22)) ?
          offsetof(CpuContext, hot) + offsetof(JitHot, spsr) : ds::jit::OFF_CPSR;
      e.lw(T1, status, R_CTX); e.nop();
      putr(e, (is[q] >> 12) & 15u, T1, cache_mask);
      for (size_t p : skip) e.patch_branch(p, e.size());
      e.addiu(R_GUEST_PC, R_GUEST_PC, 4);
      ex.push_back(e.blez(R_BUDGET)); e.nop();
      ++native_count;
      rt().stats.instrs_translated++;
      ++q;
      continue;
    }
    if (use_native && !thumb &&
        (!native_limit || native_count < native_limit) && native_msr_flags(is[q])) {
      note_code_dep(b, pc + 8);
      emit_timing(e, *jc.ctx, pc, false);
      std::vector<size_t> skip;
      emit_skip_if_false(e, is[q] >> 28, skip);
      if (ds::arm::decode_arm(is[q]) == AOp::MsrImm)
        li(e, T1, ai(is[q]));
      else
        getr(e, T1, is[q] & 15u, cache_mask);
      e.lw(T0, ds::jit::OFF_CPSR, R_CTX); e.nop();
      li(e, T2, 0x00ffffffu); e.and_(T0, T0, T2);
      li(e, T2, 0xff000000u); e.and_(T1, T1, T2);
      e.or_(T0, T0, T1);
      e.sw(T0, ds::jit::OFF_CPSR, R_CTX);
      for (size_t p : skip) e.patch_branch(p, e.size());
      e.addiu(R_GUEST_PC, R_GUEST_PC, 4);
      ex.push_back(e.blez(R_BUDGET)); e.nop();
      ++native_count;
      rt().stats.instrs_translated++;
      ++q;
      continue;
    }
    if (use_native && !thumb &&
        (!native_limit || native_count < native_limit) && native_msr_control(is[q])) {
      note_code_dep(b, pc + 8);
      emit_arm_msr_control(e, *jc.ctx, pc, is[q], cache_mask, ex);
      ++native_count;
      rt().stats.instrs_translated++;
      ++q;
      continue;
    }
    if (use_native && !thumb &&
        (!native_limit || native_count < native_limit) && native_msr_spsr(is[q])) {
      note_code_dep(b, pc + 8);
      emit_timing(e, *jc.ctx, pc, false);
      std::vector<size_t> skip;
      emit_skip_if_false(e, is[q] >> 28, skip);
      e.lw(T0, ds::jit::OFF_CPSR, R_CTX); e.nop();
      e.andi(T0, T0, 31); e.addiu(T2, 0, 0x10);
      const size_t no_spsr = e.beq(T0, T2); e.nop();
      e.addiu(T2, 0, 0x1f);
      const size_t no_spsr2 = e.beq(T0, T2); e.nop();
      if (ds::arm::decode_arm(is[q]) == AOp::MsrImm) li(e, T1, ai(is[q]));
      else getr(e, T1, is[q] & 15u, cache_mask);
      const u32 fields = (is[q] >> 16) & 15u;
      const u32 mask = ((fields & 1) ? 0x000000ffu : 0) |
                       ((fields & 2) ? 0x0000ff00u : 0) |
                       ((fields & 4) ? 0x00ff0000u : 0) |
                       ((fields & 8) ? 0xff000000u : 0);
      e.lw(T0, offsetof(CpuContext, hot) + offsetof(JitHot, spsr), R_CTX); e.nop();
      li(e, T2, ~mask); e.and_(T0, T0, T2);
      li(e, T2, mask); e.and_(T1, T1, T2);
      e.or_(T0, T0, T1);
      e.sw(T0, offsetof(CpuContext, hot) + offsetof(JitHot, spsr), R_CTX);
      const size_t done = e.size();
      e.patch_branch(no_spsr, done); e.patch_branch(no_spsr2, done);
      for (size_t p : skip) e.patch_branch(p, done);
      e.addiu(R_GUEST_PC, R_GUEST_PC, 4);
      ex.push_back(e.blez(R_BUDGET)); e.nop();
      ++native_count;
      rt().stats.instrs_translated++;
      ++q;
      continue;
    }
    if (use_native_mem && !thumb &&
        (!native_limit || native_count < native_limit) && native_arm_block(is[q])) {
      note_code_dep(b, pc + 8);
      b.dep_overflow = true;
      emit_arm_block_mem(e, *jc.ctx, pc, is[q], cache_mask, ex);
      ++native_count;
      rt().stats.instrs_translated++;
      ++q;
      continue;
    }
    ThumbMem mem{};
    bool native_mem = thumb ? native_thumb_mem(is[q], mem) :
                             native_arm_mem(is[q], pc, mem, arm9);
    if (!native_mem && !thumb)
      native_mem = native_arm_single_multi(is[q], mem);
    const int slot7 = native_mem && !arm9 ?
        arm7_data_slot(*jc.ctx, pc, thumb, mem.width, mem.load) : -1;
    if (!arm9 && slot7 < 0) native_mem = false;
    if (use_native_mem && (!native_limit || native_count < native_limit) && native_mem) {
      if (native_trace)
        std::fprintf(stderr,
                     "[mips-jit] mem pc=%08x ins=%08x %s r%u->r%u off=%c%u wb=%u\n",
                     pc, is[q], mem.load ? "load" : "store", mem.rn, mem.rd,
                     mem.neg ? '-' : '+', mem.imm, mem.writeback);
      note_code_dep(b, pc + (thumb ? 4 : 8));
      b.dep_overflow = true; // the access-time row can vary with the guest address
      emit_timing(e, *jc.ctx, pc, thumb);
      std::vector<size_t> pred_skip;
      if (!thumb) emit_skip_if_false(e, mem.cond, pred_skip);
      emit_thumb_mem(e, *jc.ctx, pc, thumb, mem, cache_mask, ex, slot7);
      if (!thumb && !pred_skip.empty()) {
        const size_t executed = e.b(); e.nop();
        for (size_t p : pred_skip) e.patch_branch(p, e.size());
        e.addiu(R_GUEST_PC, R_GUEST_PC, 4);
        ex.push_back(e.blez(R_BUDGET)); e.nop();
        e.patch_branch(executed, e.size());
      }
      ++native_count;
      rt().stats.instrs_translated++;
      ++q;
      continue;
    }
    if (use_native && arm9 && thumb &&
        (!native_limit || native_count < native_limit) &&
        (is[q] & 0xf800u) == 0x4800u) {
      const u32 literal = ((pc + 4) & ~3u) + ((is[q] & 0xffu) << 2);
      u8 *p = jc.ctx->page_table.read_ptr(literal);
      if (p && (literal >> 12) == (start >> 12)) {
        N n{};
        n.op = 13;
        n.rd = (is[q] >> 8) & 7;
        n.im = true;
        n.imm = 0;
        std::memcpy(&n.imm, p, sizeof n.imm);
        note_code_dep(b, literal);
        note_data_dep(b, literal);
        emit_timing(e, *jc.ctx, pc, true);
        emit_n(e, n, cache_mask);
        const u32 nc = arm9_code_cost(*jc.ctx, pc, true);
        const u32 nd = jc.ctx->timing9[literal >> 12][2];
        const s32 combined = static_cast<s32>(nc) + static_cast<s32>(nd) - 6;
        const u32 total = combined > static_cast<s32>(nc) && combined > static_cast<s32>(nd) ?
                          static_cast<u32>(combined) : nc > nd ? nc : nd;
        if (total > nc) e.addiu(R_BUDGET, R_BUDGET, -static_cast<s32>(total - nc));
        e.addiu(R_GUEST_PC, R_GUEST_PC, 2);
        ex.push_back(e.blez(R_BUDGET));
        e.nop();
        e.nop();
        ++native_count;
        rt().stats.instrs_translated++;
        ++q;
        continue;
      }
    }
    N n{};
    if (use_native && (!native_limit || native_count < native_limit) &&
        native_insn(is[q], thumb, arm9, n)) {
      if (native_trace)
        std::fprintf(stderr,
                     "[mips-jit] native pc=%08x ins=%08x op=%u rd=%u rn=%u rm=%u %s\n",
                     key_pc(ks[q]), is[q], n.op, n.rd, n.rn, n.rm,
                     n.im ? "imm" : "reg");
      note_code_dep(b, key_pc(ks[q]) + (thumb ? 4 : 8));
      emit_timing(e, *jc.ctx, key_pc(ks[q]), thumb, n.reg_shift || n.mul_arm7);
      std::vector<size_t> skip;
      emit_skip_if_false(e, n.cond, skip);
      emit_n(e, n, cache_mask);
      if (n.mul_arm7 && n.op == 16) {
        getr(e, T0, n.rn, cache_mask);
        e.sra(T1, T0, 31); e.xor_(T0, T0, T1);
        e.addiu(T3, 0, 1);
        li(e, T1, 0xffffff00u); e.and_(T2, T0, T1);
        const size_t one = e.beq(T2, 0); e.nop();
        e.addiu(T3, 0, 2);
        li(e, T1, 0xffff0000u); e.and_(T2, T0, T1);
        const size_t two = e.beq(T2, 0); e.nop();
        e.addiu(T3, 0, 3);
        li(e, T1, 0xff000000u); e.and_(T2, T0, T1);
        const size_t three = e.beq(T2, 0); e.nop();
        e.addiu(T3, 0, 4);
        if (n.accumulate) e.addiu(T3, T3, 1);
        const size_t done = e.b(); e.nop();
        e.patch_branch(one, e.size());
        if (n.accumulate) e.addiu(T3, T3, 1);
        const size_t done_one = e.b(); e.nop();
        e.patch_branch(two, e.size());
        if (n.accumulate) e.addiu(T3, T3, 1);
        const size_t done_two = e.b(); e.nop();
        e.patch_branch(three, e.size());
        if (n.accumulate) e.addiu(T3, T3, 1);
        e.patch_branch(done, e.size());
        e.patch_branch(done_one, e.size());
        e.patch_branch(done_two, e.size());
        e.subu(R_BUDGET, R_BUDGET, T3);
      } else if (n.reg_shift || n.op == 16 || n.op == 17) {
        e.addiu(R_BUDGET, R_BUDGET,
                -(n.setflags && (n.op == 16 || n.op == 17) ? 3 : 1));
      }
      for (size_t p : skip) e.patch_branch(p, e.size());
      e.addiu(R_GUEST_PC, R_GUEST_PC, thumb ? 2 : 4);
      // Native timing must leave at the same boundary as the interpreter:
      // budget exhaustion at exactly zero is already out of budget.
      ex.push_back(e.blez(R_BUDGET));
      e.nop();
      // 74Kc/XBurst can mis-handle back-to-back generated load/store groups.
      // Keep a real instruction boundary between native guest operations.
      e.nop();
      ++native_count;
      rt().stats.instrs_translated++;
      ++q;
      continue;
    }
    if (use_native && thumb && (!native_limit || native_count < native_limit) &&
        native_thumb_special(is[q], key_pc(ks[q]), n)) {
      note_code_dep(b, key_pc(ks[q]) + 4);
      emit_timing(e, *jc.ctx, key_pc(ks[q]), true);
      emit_n(e, n, cache_mask);
      e.addiu(R_GUEST_PC, R_GUEST_PC, 2);
      ex.push_back(e.blez(R_BUDGET));
      e.nop();
      e.nop();
      ++native_count;
      rt().stats.instrs_translated++;
      ++q;
      continue;
    }
    B x{};
    u32 z = q;
    auto native_boundary = [&](u32 i) {
      if (use_native && (!native_limit || native_count < native_limit) &&
          (native_insn(is[i], thumb, arm9, n) ||
           (thumb && native_thumb_special(is[i], key_pc(ks[i]), n))))
        return true;
      if (use_native && !thumb && (!native_limit || native_count < native_limit) &&
          (native_ignored_mcr(is[i], arm9) || native_mrs_cpsr(is[i]) ||
           native_mrs_spsr(is[i]) ||
           native_msr_flags(is[i]) || native_msr_control(is[i]) ||
           native_msr_spsr(is[i])))
        return true;
      if (use_native_branch && (!native_limit || native_count < native_limit)) {
        if (!thumb && native_arm_exc_return(is[i])) return true;
        if (!thumb && (arm_branch(is[i]) ||
            ((is[i] & 0x0FFFFFF0u) == 0x012FFF10u && (is[i] & 15u) != 15u)))
          return true;
        if (thumb && (thumb_branch(is[i]) ||
            (thumb_bl_prefix(is[i]) && i + 1 < is.size() &&
             thumb_bl_suffix(is[i + 1]))))
          return true;
      }
      if (use_native_mem && (!native_limit || native_count < native_limit)) {
        if (!thumb && native_arm_block(is[i])) return true;
        ThumbMem m{};
        bool candidate = thumb ? native_thumb_mem(is[i], m) :
            native_arm_mem(is[i], key_pc(ks[i]), m, arm9);
        if (!candidate && !thumb) candidate = native_arm_single_multi(is[i], m);
        if (candidate && (arm9 || arm7_data_slot(*jc.ctx, key_pc(ks[i]), thumb,
                                                   m.width, m.load) >= 0))
          return true;
      }
      return false;
    };
    while (q < is.size() && !native_boundary(q)) {
      x.i.push_back(is[q]);
      x.k.push_back(ks[q++]);
    }
    x.ip = e.size();
    e.lui(5, 0);
    e.ori(5, 5, 0);
    x.kp = e.size();
    e.lui(6, 0);
    e.ori(6, 6, 0);
    e.addiu(7, 0, q - z);
    e.move(4, R_CTX);
    e.load_ptr(25, reinterpret_cast<const void *>(&ds_jit_mips_fallback_block));
    if (cache_regs)
      spill_cached(e, cache_mask);
    if (has_native) e.sw(R_GUEST_PC, OR + 60, R_CTX);
    if (has_native) e.sw(R_BUDGET, OB, R_CTX);
    e.jalr(R_RA, 25);
    e.nop();
    if (has_native) {
      e.lw(R_BUDGET, OB, R_CTX);
      e.nop();
      e.lw(R_GUEST_PC, OR + 60, R_CTX);
      e.nop();
    }
    if (cache_regs)
      reload_cached(e, cache_mask);
    x.br = e.bnez(2);
    e.nop();
    bs.push_back(std::move(x));
  }
  const size_t done = e.size();
  if (has_native) e.sw(R_GUEST_PC, OR + 60, R_CTX);
  if (has_native) e.sw(R_BUDGET, OB, R_CTX);
  if (cache_regs)
    spill_cached(e, cache_mask);
  for (size_t p : ex)
    e.patch_branch(p, done);
  for (size_t p : bx)
    e.patch_branch(p, done);
  for (B &x : bs)
    e.patch_branch(x.br, done);
  e.lw(R_CTX, 16, R_SP);
  e.lw(R_RA, 20, R_SP);
  if (has_native) {
    e.lw(R_BUDGET, 48, R_SP);
    e.lw(R_FP, 56, R_SP);
  }
  if (cache_regs) {
    for (u32 r = 0; r < 15; ++r) {
      if (!(cache_mask & (1u << r))) continue;
      const u32 slot = __builtin_popcount(cache_mask & ((1u << r) - 1));
      e.lw(CACHE_FIRST + slot, CACHE_SAVED + 4 * slot, R_SP);
      e.nop();
    }
  }
  e.addiu(R_SP, R_SP, static_cast<s32>(has_native ? CACHE_FRAME : 24));
  e.jr(R_RA);
  e.nop();
  for (B &x : bs) {
    while (e.size() & 3)
      e.nop();
    size_t ip = e.size();
    for (u32 v : x.i)
      e.w(v);
    size_t kp = e.size();
    for (u32 v : x.k)
      e.w(v);
    e.patch_ptr(x.ip, 5, buf + ip);
    e.patch_ptr(x.kp, 6, buf + kp);
  }
  size = e.size();
  b.guest_len = addr - start;
  b.hot_size = size;
  b.npages = 1;
  b.dead = false;
  u8 *h = jc.ctx->page_table.read_ptr(start);
  b.host_pages[0] = reinterpret_cast<const u8 *>(
      reinterpret_cast<uintptr_t>(h) & ~uintptr_t{4095});
  b.host_lo = h;
  b.host_hi = h + b.guest_len - 1;
  return true;
}
} // namespace ds::jit::backend
