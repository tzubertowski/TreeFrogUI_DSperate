#include "core/cpu/arm_decode.h"
#include "core/cpu/jit/jit_internal.h"
#include "core/cpu/jit/mips/emit.h"
#include "core/mem/timing.h"
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <vector>

extern "C" ds::u32 ds_jit_mips_fallback_block(ds::CpuContext *, const ds::u32 *,
                                              const ds::u32 *, ds::u32);
namespace {
using ds::CpuContext;
using ds::s32;
using ds::u32;
using ds::arm::AOp;
using ds::jit::MipsEmitter;
constexpr u32 R_CTX = 16, R_SP = 29, R_RA = 31, T0 = 8, T1 = 9, T2 = 10;
constexpr u32 OR = ds::jit::OFF_REGS, OB = ds::jit::OFF_BUDGET;
struct N {
  u32 op, rd, rn, rm, imm, shift, cond;
  bool im, setflags, carry_valid, carry;
  ds::u8 shift_type;
};
static u32 ai(u32 x) {
  u32 v = x & 255, n = ((x >> 8) & 15) * 2;
  return n ? (v >> n) | (v << (32 - n)) : v;
}
static bool native(u32 x, N &n, bool shifts) {
  AOp a = ds::arm::decode_arm(x);
  n.cond = x >> 28;
  if (n.cond != 14 || (a != AOp::DpImm && a != AOp::DpImmShift))
    return false;
  n.op = (x >> 21) & 15;
  const char *op_env = std::getenv("DS_MIPS_NATIVE_OP");
  if (op_env && n.op != static_cast<u32>(std::strtoul(op_env, nullptr, 10)))
    return false;
  n.rd = (x >> 12) & 15;
  n.rn = (x >> 16) & 15;
  n.rm = x & 15;
  n.setflags = x & (1u << 20);
  if (n.rd == 15 || n.rn == 15 || n.rm == 15)
    return false;
  if (n.op != 0 && n.op != 1 && n.op != 4 && n.op != 12 &&
      n.op != 14 && n.op != 15)
    return false;
  // Keep CPSR updates in the interpreter until the hardware path is proven.
  if (n.setflags)
    return false;
  if (a == AOp::DpImm) {
    n.im = true;
    n.imm = ai(x);
    n.carry_valid = ((x >> 8) & 15) != 0;
    n.carry = n.imm >> 31;
    return true;
  }
  if ((x & 0x10) || (!shifts && ((x >> 4) & 255)))
    return false;
  n.im = false;
  n.shift = (x >> 7) & 31;
  n.shift_type = (x >> 5) & 3;
  return n.shift_type == 0 ||
         (n.shift && (n.shift_type == 1 || n.shift_type == 2));
}
static void li(MipsEmitter &e, u32 r, u32 v) {
  e.lui(r, v >> 16);
  e.ori(r, r, v);
}
static void emit_arm9_timing(MipsEmitter &e, const CpuContext &c, u32 pc) {
  const ds::u8 t = c.timing9[(pc + 8) >> 12][0];
  const u32 cost = t == 255 ? (!((pc + 8) & 31u) ? 3 : 1) : t;
  li(e, T0, pc + 8);
  e.sw(T0, OR + 60, R_CTX);
  e.lw(T0, OB, R_CTX);
  e.addiu(T0, T0, -static_cast<s32>(cost));
  e.sw(T0, OB, R_CTX);
}
static void getr(MipsEmitter &e, u32 d, u32 r) {
  e.lw(d, OR + 4 * r, R_CTX);
}
static void putr(MipsEmitter &e, u32 r, u32 s) {
  e.sw(s, OR + 4 * r, R_CTX);
}
static void note_code_dep(ds::jit::Block &b, u32 addr) {
  const u32 page = addr >> 12;
  for (u32 i = 0; i < b.ndep; ++i)
    if (b.dep_page[i] == page) {
      b.dep_kind[i] |= ds::mem::Timing::RETIME_CODE;
      return;
    }
  if (b.ndep < ds::jit::Block::DEP_MAX) {
    b.dep_page[b.ndep] = page;
    b.dep_kind[b.ndep++] = ds::mem::Timing::RETIME_CODE;
  } else
    b.dep_overflow = true;
}
static void emit_n(MipsEmitter &e, const N &n) {
  getr(e, T0, n.rn);
  if (n.im)
    li(e, T1, n.imm);
  else {
    getr(e, T1, n.rm);
    if (n.shift) {
      if (n.shift_type == 1)
        e.srl(T1, T1, n.shift);
      else if (n.shift_type == 2)
        e.sra(T1, T1, n.shift);
      else
        e.sll(T1, T1, n.shift);
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
    e.subu(T2, T0, T1);
    break;
  case 4:
    // Some 74Kc/XBurst revisions mis-execute back-to-back generated ADDU.
    // Equivalent modulo-32-bit form keeps ADD native without that opcode.
    e.nor(T1, T1, 0);
    e.subu(T2, T0, T1);
    e.addiu(T2, T2, -1);
    break;
  case 12:
    e.or_(T2, T0, T1);
    break;
  case 13:
    e.move(T2, T1);
    break;
  case 14:
    e.nor(T1, T1, 0);
    e.and_(T2, T0, T1);
    break;
  default:
    e.nor(T2, T1, 0);
  }
  if (n.setflags) {
    e.lw(T1, ds::jit::OFF_CPSR, R_CTX);
    li(e, T0, n.carry_valid ? 0x1fffffffu : 0x3fffffffu);
    e.and_(T1, T1, T0);
    e.srl(T0, T2, 31);
    e.sll(T0, T0, 31);
    e.or_(T1, T1, T0);
    size_t nz = e.bnez(T2);
    e.nop();
    li(e, T0, 0x40000000u);
    e.or_(T1, T1, T0);
    e.patch_branch(nz, e.size());
    if (n.carry_valid && n.carry) {
      li(e, T0, 0x20000000u);
      e.or_(T1, T1, T0);
    }
    e.sw(T1, ds::jit::OFF_CPSR, R_CTX);
  }
  putr(e, n.rd, T2);
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
  const bool use_native = arm9 && native_env && std::strcmp(native_env, "0");
  const char *limit_env = std::getenv("DS_MIPS_NATIVE_LIMIT");
  const u32 native_limit = limit_env ? std::strtoul(limit_env, nullptr, 10) : 0;
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
  MipsEmitter e(buf, cap);
  e.addiu(R_SP, R_SP, -24);
  e.sw(R_CTX, 16, R_SP);
  e.sw(R_RA, 20, R_SP);
  e.move(R_CTX, 4);
  std::vector<B> bs;
  std::vector<size_t> ex;
  for (u32 q = 0; q < is.size();) {
    N n{};
    if (use_native && (!native_limit || native_count < native_limit) &&
        native(is[q], n, true)) {
      note_code_dep(b, key_pc(ks[q]) + 8);
      emit_arm9_timing(e, *jc.ctx, key_pc(ks[q]));
      size_t skip = 0;
      if (n.cond != 14) {
        e.lw(T0, ds::jit::OFF_CPSR, R_CTX);
        e.srl(T0, T0, 30);
        e.andi(T0, T0, 1);
        skip = n.cond == 0 ? e.beq(T0, 0) : e.bnez(T0);
        e.nop();
      }
      emit_n(e, n);
      if (skip)
        e.patch_branch(skip, e.size());
      e.lw(T0, OR + 60, R_CTX);
      e.addiu(T0, T0, 4);
      e.sw(T0, OR + 60, R_CTX);
      e.lw(T0, OB, R_CTX);
      ex.push_back(e.bltz(T0));
      e.nop();
      // 74Kc/XBurst can mis-handle back-to-back generated load/store groups.
      // Keep a real instruction boundary between native guest operations.
      e.nop();
      ++native_count;
      ++q;
      continue;
    }
    B x{};
    u32 z = q;
    while (q < is.size() &&
           !(use_native && (!native_limit || native_count < native_limit) &&
             native(is[q], n, true))) {
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
    e.jalr(R_RA, 25);
    e.nop();
    x.br = e.bnez(2);
    e.nop();
    bs.push_back(std::move(x));
  }
  const size_t done = e.size();
  for (size_t p : ex)
    e.patch_branch(p, done);
  for (B &x : bs)
    e.patch_branch(x.br, done);
  e.lw(R_CTX, 16, R_SP);
  e.lw(R_RA, 20, R_SP);
  e.addiu(R_SP, R_SP, 24);
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
  b.nsucc = 0;
  b.dead = false;
  u8 *h = jc.ctx->page_table.read_ptr(start);
  b.host_pages[0] = reinterpret_cast<const u8 *>(
      reinterpret_cast<uintptr_t>(h) & ~uintptr_t{4095});
  b.host_lo = h;
  b.host_hi = h + b.guest_len - 1;
  return true;
}
} // namespace ds::jit::backend
