// SPDX-License-Identifier: GPL-3.0-or-later
// Recompiler vs interpreter differential test (AArch64 only).
// Random straight-line ARM and Thumb sequences run on two machines,
// one per engine, from identical state; the registers, flags, consumed cycles
// and memory must agree afterwards. The first disagreement prints the
// sequence, which names the broken instruction.
#include <cstdlib>
#include "core/nds.h"
#include "core/cpu/arm_decode.h"
#include "core/cpu/timing_mode.h"
#include "core/cpu/interp/interp.h"
#include "core/cpu/jit/jit.h"
#include "core/mem/bus.h"
#include "check.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <utility>
#include <vector>

using namespace ds;
using ds::arm::AOp;

namespace {

constexpr u32 CODE_BASE = 0x02000000, HALT_STUB = 0x02001000, BUF_BASE = 0x02200000, STACK = 0x02300000;
#if defined(__mips__)
constexpr bool MIPS_HOST = true;
#else
constexpr bool MIPS_HOST = false;
#endif

struct Machine {
  NDS nds;
  CpuContext& cpu;
  explicit Machine(Cpu which) : cpu(nds.cpu(which)) {}
  u8* host(u32 addr) { return cpu.page_table.read_ptr(addr); }
  void poke32(u32 addr, u32 v) { std::memcpy(host(addr), &v, 4); }
  void poke16(u32 addr, u16 v) { std::memcpy(host(addr), &v, 2); }
};

// ARM halt stub: ARM9 waits for interrupt through CP15; ARM7 writes HALTCNT.
void write_halt_stub(Machine& m, bool a9) {
  // Exception vectors (empty BIOS in this harness) branch to the halt stub,
  // so an undefined instruction, SWI or IRQ ends the trial instead of
  // executing zeros until the budget runs out.
  const u32 vbase = a9 ? 0xFFFF0000u : 0u;
  for (u32 v = 0; v < 0x20; v += 4) m.poke32(vbase + v, 0xE59FF000u | (0x20 - v - 8));   // ldr pc, [pc, #k] -> pool
  m.poke32(vbase + 0x20, HALT_STUB);
  if (a9) {
    m.poke32(HALT_STUB, 0xEE070F90);        // mcr p15, 0, r0, c7, c0, 4
    m.poke32(HALT_STUB + 4, 0xEAFFFFFE);    // b . (never reached)
  } else {
    m.poke32(HALT_STUB, 0xE3A00301);        // mov r0, #0x04000000
    m.poke32(HALT_STUB + 4, 0xE2800C03);    // add r0, r0, #0x300
    m.poke32(HALT_STUB + 8, 0xE3A01080);    // mov r1, #0x80
    m.poke32(HALT_STUB + 12, 0xE5C01001);   // strb r1, [r0, #1]   (0x04000301)
    m.poke32(HALT_STUB + 16, 0xEAFFFFFE);   // b .
  }
}

struct Rng {
  std::mt19937 g;
  explicit Rng(u32 seed) : g(seed) {}
  u32 next() { return g(); }
  u32 below(u32 n) { return g() % n; }
  bool coin(u32 pct = 50) { return below(100) < pct; }
};

// ---- ARM generator ----------------------------------------------------------
// Registers: r9 = buffer base (never written), r13 = stack, r15 never written.
u32 gen_arm(Rng& r) {
  const u32 cond = r.coin(80) ? 0xE : r.below(14);
  auto reg_nw = [&]() { u32 x; do { x = r.below(15); } while (x == 9 || x == 13); return x; };   // writable
  auto reg_rd = [&]() { u32 x; do { x = r.below(16); } while (x == 9 || x == 13); return x; };   // readable (pc ok)
  for (;;) {
    switch (r.below(10)) {
    case 0: case 1: case 2: {   // data processing
      const u32 opcode = r.below(16), s = r.coin(60), rd = reg_nw(), rn = reg_rd();
      const bool test = opcode >= 8 && opcode <= 0xB;
      u32 instr = (cond << 28) | (opcode << 21) | (s << 20) | (rn << 16) | (rd << 12);
      if (test) instr |= 1u << 20;
      switch (r.below(3)) {
      case 0: instr |= (1u << 25) | (r.below(16) << 8) | r.below(256); break;                       // imm
      case 1: instr |= (r.below(32) << 7) | (r.below(4) << 5) | reg_rd(); break;                    // imm shift
      default: instr |= (reg_nw() << 8) | (r.below(4) << 5) | 0x10 | reg_rd(); break;              // reg shift
      }
      return instr;
    }
    case 3: {   // multiply
      const u32 rd = reg_nw(), rn = reg_nw(), rs = reg_nw(), rm = reg_nw();
      if (rd == 15 || rn == 15 || rs == 15 || rm == 15) continue;
      const u32 s = r.coin(50);
      switch (r.below(4)) {
      case 0: return (cond << 28) | (s << 20) | (rd << 16) | (rs << 8) | 0x90 | rm;                 // MUL
      case 1: return (cond << 28) | (1u << 21) | (s << 20) | (rd << 16) | (rn << 12) | (rs << 8) | 0x90 | rm;   // MLA
      default: {
        if (rd == rn) continue;
        const u32 op = 4 + r.below(4);   // UMULL UMLAL SMULL SMLAL
        return (cond << 28) | (op << 21) | (s << 20) | (rd << 16) | (rn << 12) | (rs << 8) | 0x90 | rm;
      }
      }
    }
    case 4: case 5: {   // LDR/STR word/byte, base r9, pre-indexed, no writeback
      const u32 l = r.coin(50), b = r.coin(30), rd = reg_rd();
      if (l && rd == 15) continue;
      u32 instr = (cond << 28) | (1u << 26) | (1u << 24) | (1u << 23) | (b << 22) | (l << 20) | (9u << 16) | (rd << 12);
      if (r.coin(70)) instr |= r.below(0x800);
      else instr |= (1u << 25) | (r.below(8) << 7) | (r.below(3) << 5) | reg_nw();   // reg offset (may wander; both engines agree)
      return instr;
    }
    case 6: {   // LDRH/STRH/LDRSB/LDRSH, base r9, immediate
      const u32 l = r.coin(50), rd = reg_nw();
      u32 sh = l ? 1 + r.below(3) : 1;
      const u32 off = r.below(256);
      return (cond << 28) | (1u << 24) | (1u << 23) | (1u << 22) | (l << 20) | (9u << 16) | (rd << 12) | ((off >> 4) << 8) | 0x90 | (sh << 5) | (off & 0xF);
    }
    case 7: {   // LDM/STM base r9, no writeback, no pc
      u32 list = r.next() & 0x5DFF;   // no r9, r13, r15
      if (!list) continue;
      const u32 l = r.coin(50), p = r.coin(50), u = r.coin(50);
      return (cond << 28) | (4u << 25) | (p << 24) | (u << 23) | (l << 20) | (9u << 16) | list;
    }
    case 8: {   // CLZ / MRS / SWP(B) [r9]
      if (r.coin(30)) return (cond << 28) | 0x01000090 | (r.coin(40) << 22) | (9u << 16) | (reg_nw() << 12) | reg_nw();
      if (r.coin(50)) return (cond << 28) | 0x016F0F10 | (reg_nw() << 12) | reg_nw();
      return (cond << 28) | 0x010F0000 | (reg_nw() << 12);
    }
    default: {  // PUSH/POP style: STMDB sp!/LDMIA sp! without pc
      u32 list = r.next() & 0x5DFF;
      if (!list) continue;
      if (r.coin(50)) return (cond << 28) | (0xE92D0000u & 0x0FFFFFFF) | list;   // stmdb sp!, {list}
      return (cond << 28) | (0xE8BD0000u & 0x0FFFFFFF) | list;                 // ldmia sp!, {list}
    }
    }
  }
}

// A self-contained `LDM ^` / `STM ^` gadget. The generator otherwise never
// leaves SYS mode, where the user bank IS the live one and the transfer is
// indistinguishable from a plain LDM/STM -- so nothing would ever reach the
// inline path's four inlinable modes, nor its FIQ/USR guard.
//
// The base is copied into r0 first because r9 (the buffer base the generator
// relies on) is banked away in FIQ; r0-r7 are not banked in any mode, so the
// transfer addresses the buffer whichever mode the gadget picked.
void gen_user_bank(Rng& r, std::vector<u32>& out) {
  static const u32 modes[] = {0x10, 0x11, 0x12, 0x13, 0x17, 0x1B, 0x1F};
  const u32 mode = modes[r.below(7)];
  const u32 load = r.coin(50);
  u32 list;
  do { list = r.next() & (load ? 0x7FFFu : 0xFFFFu); } while (!list);   // r15 in a load list is a CPSR restore, not a user-bank transfer
  out.push_back(0xE1A00009);                            // mov r0, r9
  out.push_back(0xE321F000 | (0xC0 | mode));            // msr cpsr_c, #(I|F|mode)
  out.push_back(0xE8C00000 | (load << 20) | list);      // ldm/stm r0, {list}^
  out.push_back(0xE321F0DF);                            // msr cpsr_c, #(I|F|SYS) -- a no-op from USR, which cannot leave
}

// ---- Thumb generator ---------------------------------------------------------
// r6 = buffer base (kept), sp = stack.
u16 gen_thumb(Rng& r) {
  auto lo = [&]() { u32 x; do { x = r.below(8); } while (x == 6); return x; };
  for (;;) {
    switch (r.below(12)) {
    case 0: return static_cast<u16>((r.below(3) << 11) | (r.below(32) << 6) | (lo() << 3) | lo());          // shift imm
    case 1: return static_cast<u16>(0x1800 | (r.below(4) << 9) | (r.below(8) << 6) | (lo() << 3) | lo());   // add/sub reg/imm3 (rn may be r6: read only)
    case 2: return static_cast<u16>(0x2000 | (r.below(4) << 11) | (lo() << 8) | r.below(256));             // mov/cmp/add/sub imm8
    case 3: return static_cast<u16>(0x4000 | (r.below(16) << 6) | (lo() << 3) | lo());                      // alu
    case 4: {   // hi reg ops, never pc/sp as destination, no r6
      const u32 op = r.below(3);
      u32 rd, rs;
      do { rd = r.below(15); } while (rd == 6 || rd == 13);
      do { rs = r.below(16); } while (rs == 6);
      if (op == 1 && rd == 15) continue;
      return static_cast<u16>(0x4400 | (op << 8) | ((rd >> 3) << 7) | (rs << 3) | (rd & 7));
    }
    case 5: return static_cast<u16>(0x4800 | (lo() << 8) | r.below(256));                                    // ldr pc-rel (reads code area)
    case 6: return static_cast<u16>(0x5000 | (r.below(8) << 9) | (lo() << 6) | (6u << 3) | lo());            // ldr/str reg [r6 + ro]
    case 7: return static_cast<u16>(0x6000 | (r.below(4) << 11) | (r.below(32) << 6) | (6u << 3) | lo());    // ldr/str imm5 [r6]
    case 8: return static_cast<u16>(0x8000 | (r.below(2) << 11) | (r.below(32) << 6) | (6u << 3) | lo());    // ldrh/strh
    case 9: return static_cast<u16>(0x9000 | (r.below(2) << 11) | (lo() << 8) | r.below(256));               // sp-relative
    case 10: {  // push/pop without pc/lr, adjust sp, add rd, sp/pc
      switch (r.below(3)) {
      case 0: { u32 list = r.next() & 0xBF; if (!list) continue; return static_cast<u16>(0xB400 | (r.below(2) << 11) | list); }
      case 1: return static_cast<u16>(0xB000 | (r.below(2) << 7) | r.below(16));
      default: return static_cast<u16>(0xA000 | (r.below(2) << 11) | (lo() << 8) | r.below(256));
      }
    }
    default: {  // stmia/ldmia rb!, rb != 6, list without rb
      const u32 rb = lo();
      u32 list = r.next() & 0xBF & ~(1u << rb);
      if (!list) continue;
      return static_cast<u16>(0xC000 | (r.below(2) << 11) | (rb << 8) | list);
    }
    }
  }
}

// The machines are reused across trials, and a trial stores wherever its
// registers point -- over its own code, past the ends of the buffer and stack
// windows, anywhere in RAM. What it leaves behind is the next trial's starting
// memory. That is harmless while the two engines agree, because then they
// leave the SAME residue; it stops being harmless the moment a trial ends with
// them holding different memory, after which a later seed can fail on residue
// rather than on its own behaviour. Two mismatches found while adding the
// user-bank gadget reproduced only in the full set and passed on their own,
// and one of them was residue, not a defect.
//
// So: wipe main RAM only after a trial that could have left the two sides
// different -- a mismatch, or an inconclusive run where both engines ran out
// of budget in different places. Page at a time; a word at a time is a
// page-table lookup per word and costs minutes a run under qemu.
void wipe_ram(Machine& m) {
  for (u32 a = CODE_BASE; a < CODE_BASE + mem::Bus::MAIN_RAM_SIZE; a += mem::PAGE_SIZE)
    std::memset(m.host(a), 0, mem::PAGE_SIZE);
}

struct Trial {
  bool thumb;
  std::vector<u32> code;   // ARM words or Thumb halfwords
  u32 regs[16];
  u32 cpsr;
  u32 spsr = 0;
  u32 code_base = CODE_BASE;
  s32 budget = 1 << 24;
  bool code_in_r9 = false;
  bool require_halt = false;
  bool strict_budget = false;
};

u32 g_selfmod = 0;
bool g_ram_dirty = false;   // the last trial may have left the two machines' RAM different
u8 g_code_image[0x1000];    // the code page as load_trial wrote it

bool wrote_own_code(Machine& m) { return std::memcmp(m.host(CODE_BASE), g_code_image, sizeof g_code_image) != 0; }

void load_trial(Machine& m, const Trial& t, bool a9) {
  write_halt_stub(m, a9);
  u32 addr = t.code_base;
  if (t.thumb) {
    for (u32 h : t.code) { m.poke16(addr, static_cast<u16>(h)); addr += 2; }
    // ldr r0, [pc, #k]; bx r0; (pad) .word HALT_STUB
    const u32 ldr_at = addr;
    const u32 pool = (ldr_at + 4 + 3) & ~3u;
    m.poke16(ldr_at, static_cast<u16>(0x4800 | ((pool - ((ldr_at + 4) & ~3u)) >> 2)));
    m.poke16(ldr_at + 2, 0x4700);
    if (pool > ldr_at + 4) m.poke16(ldr_at + 4, 0x46C0);   // nop
    m.poke32(pool, HALT_STUB);
  } else {
    for (u32 w : t.code) { m.poke32(addr, w); addr += 4; }
    const s32 off = static_cast<s32>(HALT_STUB - (addr + 8)) >> 2;
    m.poke32(addr, 0xEA000000u | (static_cast<u32>(off) & 0x00FFFFFF));   // b halt_stub
  }
  for (u32 i = 0; i < 0x2000; i += 4) m.poke32(BUF_BASE + i, 0x01010101u * (i >> 2) ^ 0xA5A5A5A5u);
  for (u32 i = 0; i < 0x400; i += 4) m.poke32(STACK - 0x200 + i, 0x11111111u * (i >> 2));
  CpuContext& c = m.cpu;
  c.set_cpsr(0x1F);                // SYS mode, ARM
  // The banked registers are part of the starting state, and the machines are
  // reused across trials: without this a trial that ends in a non-SYS mode
  // leaves its banks to the next one, which then depends on trial order
  // rather than on its own seed. It never mattered while nothing switched
  // mode; the user-bank gadget does.
  std::memset(c.bank_r8_r12, 0, sizeof c.bank_r8_r12);
  std::memset(c.bank_r13, 0, sizeof c.bank_r13);
  std::memset(c.bank_r14, 0, sizeof c.bank_r14);
  std::memset(c.bank_spsr, 0, sizeof c.bank_spsr);
  c.hot.spsr = 0;
  for (int i = 0; i < 15; ++i) c.hot.regs[i] = t.regs[i];
  if (t.code_in_r9) c.hot.regs[9] = t.code_base;
  c.hot.cpsr = t.cpsr | 0x1F | (t.thumb ? 0x20 : 0);
  c.hot.spsr = t.spsr;
  c.hot.regs[15] = t.code_base + (t.thumb ? 4 : 8);
  c.halted = false;
  c.hot.irq_pending = 0;
  // As if a jump had just landed at CODE_BASE (the ARM7 interpreter keeps the
  // code region of the last jump target).
  c.code_cycles = t.code_base >> 15;
  c.code_region = t.code_base >> 24;
  c.hot.cycle_budget = t.budget;
  c.budget_at_halt = 0;
  c.jumped = false;
  std::memcpy(g_code_image, m.host(CODE_BASE), sizeof g_code_image);
}

void run_machine(Machine& m, RunFn fn) {
  for (int guard = 0; guard < 100000 && !m.cpu.halted && m.cpu.hot.cycle_budget > 0; ++guard) fn(m.cpu);
}

u32 g_inconclusive = 0;

bool compare(Machine& a, Machine& b, const Trial& t, u32 seed, bool report) {
  // A trial that never reaches the halt stub on either engine (a store that
  // clobbered its own code, a loop through stale memory) only differs in
  // where the budget ran out, which is the one thing the engines are allowed
  // to differ in. Count it and move on.
  if (!a.cpu.halted && !b.cpu.halted && a.cpu.hot.cycle_budget <= 0 && b.cpu.hot.cycle_budget <= 0 && !t.require_halt && !t.strict_budget) { ++g_inconclusive; return true; }
  bool ok = true;
  if (t.require_halt && (!a.cpu.halted || !b.cpu.halted)) ok = false;
  for (int i = 0; i < 16; ++i) if (a.cpu.hot.regs[i] != b.cpu.hot.regs[i]) ok = false;
  if (a.cpu.hot.cpsr != b.cpu.hot.cpsr) ok = false;
  if (a.cpu.halted != b.cpu.halted) ok = false;
  // Both engines end the slice with budget -1 on a halt; the budget at the
  // moment of halting is the consumed-cycle comparison.
  const s32 ba = a.cpu.halted ? a.cpu.budget_at_halt : a.cpu.hot.cycle_budget;
  const s32 bb = b.cpu.halted ? b.cpu.budget_at_halt : b.cpu.hot.cycle_budget;
  if (ba != bb) ok = false;
  if (std::memcmp(a.host(BUF_BASE), b.host(BUF_BASE), 0x2000) != 0) ok = false;
  if (std::memcmp(a.host(STACK - 0x200), b.host(STACK - 0x200), 0x400) != 0) ok = false;
  if (!ok) g_ram_dirty = true;
  if (ok || !report) return ok;
  std::fprintf(stderr, "MISMATCH seed %u (%s), shortest failing prefix:\n", seed, t.thumb ? "thumb" : "arm");
  for (size_t i = 0; i < t.code.size(); ++i) std::fprintf(stderr, "  %08x: %0*x\n", t.code_base + static_cast<u32>(i * (t.thumb ? 2 : 4)), t.thumb ? 4 : 8, t.code[i]);
  std::fprintf(stderr, "  initial: cpsr %08x", t.cpsr);
  for (int i = 0; i < 15; ++i) std::fprintf(stderr, " r%d=%08x", i, t.regs[i]);
  std::fprintf(stderr, "\n  %-6s %-10s %-10s\n", "", "interp", "jit");
  for (int i = 0; i < 16; ++i) if (a.cpu.hot.regs[i] != b.cpu.hot.regs[i]) std::fprintf(stderr, "  r%-5d %08x   %08x\n", i, a.cpu.hot.regs[i], b.cpu.hot.regs[i]);
  if (a.cpu.hot.cpsr != b.cpu.hot.cpsr) std::fprintf(stderr, "  cpsr   %08x   %08x\n", a.cpu.hot.cpsr, b.cpu.hot.cpsr);
  if (ba != bb) std::fprintf(stderr, "  budget %08x   %08x (consumed %d vs %d)\n", ba, bb, (1 << 24) - ba, (1 << 24) - bb);
  if (a.cpu.halted != b.cpu.halted) std::fprintf(stderr, "  halted %d %d\n", a.cpu.halted, b.cpu.halted);
  for (u32 i = 0; i < 0x2000; i += 4) {
    u32 x, y; std::memcpy(&x, a.host(BUF_BASE + i), 4); std::memcpy(&y, b.host(BUF_BASE + i), 4);
    if (x != y) { std::fprintf(stderr, "  buf[%04x] %08x %08x\n", i, x, y); }
  }
  // The code region is not part of the comparison (a trial that overwrites its
  // own code is allowed to end differently), but when something else differs
  // it is usually the cause, so report it.
  for (u32 i = 0; i < 0x1000; i += 4) {
    u32 x, y; std::memcpy(&x, a.host(CODE_BASE + i), 4); std::memcpy(&y, b.host(CODE_BASE + i), 4);
    if (x != y) { std::fprintf(stderr, "  code[%04x] %08x %08x\n", i, x, y); }
  }
  return false;
}

// DS_FUZZ_TRACE=<path-prefix>: dump a per-instruction trace of both engines.
// Debug scaffolding for localising a mismatch -- the comparison itself only
// looks at the end state, which names the symptom and not the instruction.
std::FILE* g_trace_fp = nullptr;
void trace_cb(ds::CpuContext& c, u32 instr, void*) {
  if (!g_trace_fp) return;
  std::fprintf(g_trace_fp, "%08x %08x cpsr=%08x budget=%d", c.hot.regs[15], instr, c.hot.cpsr, c.hot.cycle_budget);
  for (int i = 0; i < 15; ++i) std::fprintf(g_trace_fp, " %08x", c.hot.regs[i]);
  std::fprintf(g_trace_fp, " | b13=%08x b14=%08x\n", c.bank_r13[0], c.bank_r14[0]);
}

bool run_both(Machine& mi, Machine& mj, const Trial& tr, bool a9, u32 seed, bool report) {
  const char* tp = std::getenv("DS_FUZZ_TRACE");
  if (g_ram_dirty) { wipe_ram(mi); wipe_ram(mj); g_ram_dirty = false; }
  jit::flush(mj.cpu);
  load_trial(mi, tr, a9);
  load_trial(mj, tr, a9);
  if (tp) {
    std::string base = std::string(tp);
    g_trace_fp = std::fopen((base + ".interp").c_str(), "w");
    mi.nds.trace = &trace_cb;
    run_machine(mi, &interp::run);
    std::fclose(g_trace_fp);
    mi.nds.trace = nullptr;
    g_trace_fp = std::fopen((base + ".jit").c_str(), "w");
    mj.nds.trace = &trace_cb;
    jit::set_trace(true);
    jit::flush(mj.cpu);
    run_machine(mj, &jit::run);
    jit::set_trace(false);
    std::fclose(g_trace_fp);
    g_trace_fp = nullptr;
    mj.nds.trace = nullptr;
  } else {
    run_machine(mi, &interp::run);
    run_machine(mj, &jit::run);
  }
  return compare(mi, mj, tr, seed, report);
}

// Hand-written sequences for cases the generator reaches rarely. Registers
// follow the generator's conventions (r9/r6 = buffer, r13 = stack).
struct Directed {
  Cpu which; bool thumb; std::vector<u32> code; u32 regs[15];
  u32 code_base = CODE_BASE; s32 budget = 1 << 24; bool code_in_r9 = false;
  bool require_halt = false; bool strict_budget = false;
};

void directed(u32 start = 0, u32 limit = ~0u) {
  const Directed cases[] = {
    // A branch as the first instruction must leave the translated entry with
    // the branch target, not the fall-through PC.
    {Cpu::ARM9, false, {0xEA000001, 0xE2800001, 0xE2800002, 0xE2800003}, {0}, CODE_BASE, 1 << 24, false, true},
    {Cpu::ARM7, false, {0xEA000001, 0xE2800001, 0xE2800002, 0xE2800003}, {0}, CODE_BASE, 1 << 24, false, true},
    // BL enters a second block and BX LR returns to the caller's block, which
    // then skips over the inline function to the generated halt branch.
    {Cpu::ARM9, false, {0xEB000001, 0xEA000002, 0xE1A00000, 0xE2800001, 0xE12FFF1E}, {0}, CODE_BASE, 1 << 24, false, true},
    {Cpu::ARM7, false, {0xEB000001, 0xEA000002, 0xE1A00000, 0xE2800001, 0xE12FFF1E}, {0}, CODE_BASE, 1 << 24, false, true},
    // ARM9 BLX immediate exercises the interpreter fallback's mixed-state transfer.
    {Cpu::ARM9, false, {0xFA000000, 0xE2800001, 0x46C04708}, {0, HALT_STUB}, CODE_BASE, 1 << 24, false, true},
    // Return through the LR written by BLX, then continue in ARM state.
    {Cpu::ARM9, false, {0xFA000000, 0xE1A0F001, 0x46C04770}, {0, HALT_STUB}, CODE_BASE, 1 << 24, false, true},
    // H=1 selects the halfword at +10 rather than +8.
    {Cpu::ARM9, false, {0xFB000000, 0xE2800001, 0x470846C0}, {0, HALT_STUB}, CODE_BASE, 1 << 24, false, true},
    {Cpu::ARM9, false, {0xFB000000, 0xE1A0F001, 0x477046C0}, {0, HALT_STUB}, CODE_BASE, 1 << 24, false, true},
    // Translate across a 4 KiB guest page boundary; the trailing halt branch
    // starts on the next page.
    {Cpu::ARM9, false, {0xE3A00001, 0xE2800002}, {0}, CODE_BASE + 0x2ff8, 1 << 24, false, true},
    {Cpu::ARM7, false, {0xE3A00001, 0xE2800002}, {0}, CODE_BASE + 0x2ff8, 1 << 24, false, true},
    // A small slice budget must stop the same way in both engines.
    {Cpu::ARM9, false, {0xE2800001, 0xE2800001, 0xE2800001, 0xE2800001, 0xE2800001}, {0}, CODE_BASE, 32, false, false, MIPS_HOST},
    {Cpu::ARM7, false, {0xE2800001, 0xE2800001, 0xE2800001, 0xE2800001, 0xE2800001}, {0}, CODE_BASE, 32, false, false, MIPS_HOST},
    // Store a branch over the second instruction, then loop back. This forces
    // the translated code page to invalidate and be rebuilt before halting.
    {Cpu::ARM9, false, {0xE3A00000, 0xE5892004, 0xEAFFFFFD}, {0, 0, 0xEA000000}, CODE_BASE, 1 << 24, true, true},
    {Cpu::ARM7, false, {0xE3A00000, 0xE5892004, 0xEAFFFFFD}, {0, 0, 0xEA000000}, CODE_BASE, 1 << 24, true, true},
    // Pending fetch cycles of the MOV must survive the LDR taking its slow path
    // (unmapped address through a register offset).
    {Cpu::ARM9, false, {0xE3A00001, 0xE7991102}, {0, 0, 0x12345678, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
    {Cpu::ARM7, false, {0xE3A00001, 0xE7991102}, {0, 0, 0x12345678, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
    // Same with a store, and with a conditional instruction in front.
    {Cpu::ARM9, false, {0xE0811002, 0xE7891102}, {0, 1, 0x12345678, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
    {Cpu::ARM9, false, {0x03A00001, 0xE7991102}, {0, 0, 0x12345678, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
    // CP15 cache maintenance (ignored by the core): drain write buffer,
    // invalidate I-cache line, clean+invalidate D-cache line; then an ALU op.
    {Cpu::ARM9, false, {0xEE070F9A, 0xEE073F35, 0xEE070F3E, 0xE2800001}, {5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
    // MSR CPSR_c keeping the mode (SYS): set F, then clear it; flags field from a register and an immediate.
    {Cpu::ARM9, false, {0xE129F001, 0xE129F002, 0xE128F003, 0xE328F20F, 0xE2800001}, {0, 0x5F, 0x1F, 0xF0000000u, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
    {Cpu::ARM7, false, {0xE129F001, 0xE129F002, 0xE128F003, 0xE328F20F, 0xE2800001}, {0, 0x5F, 0x1F, 0xF0000000u, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
    // MSR CPSR_c changing the mode (SYS -> IRQ -> SYS): the interpreter path; r13/r14 are banked.
    {Cpu::ARM9, false, {0xE129F001, 0xE1A0D004, 0xE129F002, 0xE2800001}, {0, 0x92, 0x1F, 0, 0x12345678, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
    // MSR CPSR_cxsf with the full mask, same mode, and an IRQ-enable (I cleared) with no IRQ pending.
    {Cpu::ARM9, false, {0xE12FF001, 0xE2800001}, {0, 0x600000DFu, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
    {Cpu::ARM9, false, {0xE129F001, 0xE2800001}, {0, 0x1F, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
    // v5TE DSP multiplies (ARM9 only). Operands are chosen so the halves are
    // distinguishable and the accumulate overflows, which is the only way the
    // sticky Q flag is exercised. r6/r9/r13 are reserved by the harness.
    // SMULBB / SMULTT / SMULBT r0, r1, r2
    {Cpu::ARM9, false, {0xE1600281, 0xE2800001}, {0, 0x7FFF8000u, 0x00028001u, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
    {Cpu::ARM9, false, {0xE16002E1, 0xE2800001}, {0, 0x7FFF8000u, 0x00028001u, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
    {Cpu::ARM9, false, {0xE16002C1, 0xE2800001}, {0, 0x7FFF8000u, 0x00028001u, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
    // SMLABB r0, r1, r2, r3 -- no overflow, then with r3 forcing Q
    {Cpu::ARM9, false, {0xE1003281, 0xE2800001}, {0, 0x00001234u, 0x00005678u, 0x100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
    {Cpu::ARM9, false, {0xE1003281, 0xE2800001}, {0, 0x7FFF7FFFu, 0x7FFF7FFFu, 0x7FFFFFFFu, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
    // SMLAWB / SMLAWT r0, r1, r2, r3, the GSDD shape, with and without overflow
    {Cpu::ARM9, false, {0xE1203281, 0xE2800001}, {0, 0x12345678u, 0x00007FFFu, 0x100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
    {Cpu::ARM9, false, {0xE12032C1, 0xE2800001}, {0, 0x80000000u, 0x8000FFFFu, 0x7FFFFFFFu, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
    // SMULWB / SMULWT r0, r1, r2
    {Cpu::ARM9, false, {0xE12002A1, 0xE2800001}, {0, 0x12345678u, 0x00007FFFu, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
    {Cpu::ARM9, false, {0xE12002E1, 0xE2800001}, {0, 0x80000000u, 0x8000FFFFu, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
    // Destination aliasing a source, and a conditional form (the smlabblo GSDD executes)
    {Cpu::ARM9, false, {0xE1013281, 0xE2800001}, {0, 0x00001234u, 0x00005678u, 0x100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
    {Cpu::ARM9, false, {0x31013281, 0xE2800001}, {0, 0x00001234u, 0x00005678u, 0x100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
    // The exact GSDD encodings: smlawb lr,fp,r2,lr and smlabblo r1,ip,r1,r2
    {Cpu::ARM9, false, {0xE12EE28B, 0xE2800001}, {0, 0, 0x00001234u, 0, 0, 0, 0, 0, 0, 0, 0, 0x00098765u, 0, 0, 0x00000042u}},
    {Cpu::ARM9, false, {0x3101218C, 0xE2800001}, {0, 0x00004321u, 0x00000100u, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x00001234u, 0, 0}},
    // On the ARM7 these encodings are undefined and must still fall back.
    {Cpu::ARM7, false, {0xE1003281, 0xE2800001}, {0, 0x00001234u, 0x00005678u, 0x100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
    // A conditional register-shift data-processing instruction: taken (CC with C clear),
    // not taken (CS), unconditional; and a taken conditional immediate form.
    {Cpu::ARM9, false, {0x31dceb70, 0xE2800001}, {0x7a2fbc14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x7b20dd03, 0xe41d8363, 0, 0xd1cea9ce}},
    {Cpu::ARM9, false, {0x21dceb70, 0xE2800001}, {0x7a2fbc14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x7b20dd03, 0xe41d8363, 0, 0xd1cea9ce}},
    {Cpu::ARM9, false, {0xE1dceb70, 0xE2800001}, {0x7a2fbc14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x7b20dd03, 0xe41d8363, 0, 0xd1cea9ce}},
    {Cpu::ARM9, false, {0x33dce001, 0xE2800001}, {0x7a2fbc14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x7b20dd03, 0xe41d8363, 0, 0xd1cea9ce}},
    {Cpu::ARM7, false, {0x31dceb70, 0xE2800001}, {0x7a2fbc14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x7b20dd03, 0xe41d8363, 0, 0xd1cea9ce}},
    // RSCS with a pc operand, then a conditional on N/V (seed 2003's shape).
    {Cpu::ARM9, false, {0xE2FFA4D2, 0xB1A01002, 0xE2800001}, {0, 0, 0x12345678u, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
    {Cpu::ARM9, false, {0xE2F0A4D2, 0xB1A01002, 0xE2800001}, {0x02000008u, 0, 0x12345678u, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
    // ... and with C set going in (cmp r0, r0 first).
    {Cpu::ARM9, false, {0xE1500000, 0xE2FFA4D2, 0xB1A01002, 0xE2800001}, {0, 0, 0x12345678u, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
    {Cpu::ARM9, false, {0xE1500000, 0xE2F0A4D2, 0xB1A01002, 0xE2800001}, {0x02000008u, 0, 0x12345678u, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
    // Indirect branches (refill costs are compared through the budget).
    // bx r1 -> the ARM halt stub; bx r1 -> Thumb code at +4 (even) / +6 (odd)
    // that loads the halt stub's address and bx's to it; blx r1 (ARM9).
    {Cpu::ARM9, false, {0xE12FFF11, 0xE2800001}, {0, HALT_STUB, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
    {Cpu::ARM7, false, {0xE12FFF11, 0xE2800001}, {0, HALT_STUB, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
    {Cpu::ARM9, false, {0xE12FFF11, 0x47004801, 0x46C046C0, HALT_STUB}, {0, CODE_BASE + 4 + 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
    {Cpu::ARM7, false, {0xE12FFF11, 0x47004801, 0x46C046C0, HALT_STUB}, {0, CODE_BASE + 4 + 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
    {Cpu::ARM9, false, {0xE12FFF11, 0x480146C0, 0x46C04700, HALT_STUB}, {0, CODE_BASE + 6 + 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
    {Cpu::ARM7, false, {0xE12FFF11, 0x480146C0, 0x46C04700, HALT_STUB}, {0, CODE_BASE + 6 + 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
    {Cpu::ARM9, false, {0xE12FFF31, 0xE2800001}, {0, HALT_STUB, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
    // Conditional bx, taken and not taken.
    {Cpu::ARM9, false, {0xE1500000, 0x012FFF11, 0xE2800001}, {0, HALT_STUB, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
    {Cpu::ARM9, false, {0xE1500000, 0x112FFF11, 0xE2800001}, {0, HALT_STUB, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
    // str r2, [r9]; ldmia r9, {pc}  and  ldmia r9!, {r0, pc} (the CDI charge after the jump)
    {Cpu::ARM9, false, {0xE5892000, 0xE8998000}, {0, 0, HALT_STUB, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
    {Cpu::ARM7, false, {0xE5892000, 0xE8998000}, {0, 0, HALT_STUB, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
    {Cpu::ARM9, false, {0xE5892004, 0xE8B98001}, {0, 0, HALT_STUB, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
    {Cpu::ARM7, false, {0xE5892004, 0xE8B98001}, {0, 0, HALT_STUB, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
    // Thumb: bx r1; mov pc, r1 / add pc, r1 (odd targets: Thumb code at +4 that bx's to the halt stub);
    // push {r2}; pop {pc}; an unpaired blx suffix (lr = halt stub).
    {Cpu::ARM9, true, {0x4708, 0x46C0, 0x4801, 0x4700, 0x46C0, 0x46C0, static_cast<u32>(HALT_STUB & 0xFFFF), static_cast<u32>(HALT_STUB >> 16)}, {0, HALT_STUB, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
    {Cpu::ARM7, true, {0x4708, 0x46C0, 0x4801, 0x4700, 0x46C0, 0x46C0, static_cast<u32>(HALT_STUB & 0xFFFF), static_cast<u32>(HALT_STUB >> 16)}, {0, HALT_STUB, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
    {Cpu::ARM9, true, {0x468F, 0x46C0, 0x4801, 0x4700, 0x46C0, 0x46C0, static_cast<u32>(HALT_STUB & 0xFFFF), static_cast<u32>(HALT_STUB >> 16)}, {0, CODE_BASE + 4 + 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
    {Cpu::ARM7, true, {0x468F, 0x46C0, 0x4801, 0x4700, 0x46C0, 0x46C0, static_cast<u32>(HALT_STUB & 0xFFFF), static_cast<u32>(HALT_STUB >> 16)}, {0, CODE_BASE + 4 + 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
    {Cpu::ARM9, true, {0x448F, 0x46C0, 0x4801, 0x4700, 0x46C0, 0x46C0, static_cast<u32>(HALT_STUB & 0xFFFF), static_cast<u32>(HALT_STUB >> 16)}, {0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
    {Cpu::ARM9, true, {0xB404, 0xBD00}, {0, 0, HALT_STUB, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
    {Cpu::ARM9, true, {0xE800}, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, HALT_STUB}},
    {Cpu::ARM9, true, {0xF000, 0xEFFE}, {}, CODE_BASE}, // paired BLX to ARM halt stub
    // SWP / SWPB [r9]: rd != rm, rd == rm (the in-place swap), rd == rn
    // (the base register takes the loaded word), conditional taken / not
    // taken, and through a register far from the buffer (the slow path).
    {Cpu::ARM9, false, {0xE5892000, 0xE1090091, 0xE2800001}, {0, 0x11111111, 0x22222222, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
    {Cpu::ARM7, false, {0xE5892000, 0xE1090091, 0xE2800001}, {0, 0x11111111, 0x22222222, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
    {Cpu::ARM9, false, {0xE5892000, 0xE1091091, 0xE2800001}, {0, 0x11111111, 0x22222222, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
    {Cpu::ARM9, false, {0xE5892000, 0xE1491091, 0xE5990000}, {0, 0x111111AB, 0x22222222, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
    {Cpu::ARM7, false, {0xE5892000, 0xE1491091, 0xE5990000}, {0, 0x111111AB, 0x22222222, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
    {Cpu::ARM9, false, {0xE1A03009, 0xE1033091, 0xE5990000}, {0, 0x11111111, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
    {Cpu::ARM9, false, {0xE1500000, 0x01090091, 0x11092091, 0xE2800001}, {0, 0x11111111, 0x22222222, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
    {Cpu::ARM9, false, {0xE1030091, 0xE2800001}, {0, 0x11111111, 0, 0x04000208, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
    {Cpu::ARM7, false, {0xE1030091, 0xE2800001}, {0, 0x11111111, 0, 0x04000208, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
    // Data-processing writes to pc (S=0): a bounds-checked jump table
    // (cmp r1, #2; addls pc, pc, r1, lsl #2) in range and out of it, an
    // unconditional add pc, pc, r2, lsl #2, add pc, pc, #0, and mov pc, r1.
    {Cpu::ARM9, false, {0xE3510002, 0x908FF101, 0xE2800064, 0xE2800001, 0xE2800002, 0xE2800004}, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
    {Cpu::ARM9, false, {0xE3510002, 0x908FF101, 0xE2800064, 0xE2800001, 0xE2800002, 0xE2800004}, {0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
    {Cpu::ARM9, false, {0xE3510002, 0x908FF101, 0xE2800064, 0xE2800001, 0xE2800002, 0xE2800004}, {0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
    {Cpu::ARM7, false, {0xE3510002, 0x908FF101, 0xE2800064, 0xE2800001, 0xE2800002, 0xE2800004}, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
    {Cpu::ARM7, false, {0xE3510002, 0x908FF101, 0xE2800064, 0xE2800001, 0xE2800002, 0xE2800004}, {0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
    {Cpu::ARM7, false, {0xE3510002, 0x908FF101, 0xE2800064, 0xE2800001, 0xE2800002, 0xE2800004}, {0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
    {Cpu::ARM9, false, {0xE08FF102, 0xE2800064, 0xE2800001, 0xE2800002}, {0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
    {Cpu::ARM9, false, {0xE28FF000, 0xE2800064, 0xE2800001}, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
    {Cpu::ARM9, false, {0xE1A0F001, 0xE2800001}, {0, HALT_STUB, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
    {Cpu::ARM7, false, {0xE1A0F001, 0xE2800001}, {0, HALT_STUB, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
    // Exit flag liveness: cmp r1, r1 (C=1); tst r0, r0 (C kept); bne -> a
    // successor that reads C first (bcs), taken and not taken, and one whose
    // successors overwrite C first (adds), where the merge is dropped.
    {Cpu::ARM9, true, {0x4289, 0x4200, 0xD100, 0x3001, 0xD200, 0x3064, 0x3001}, {5, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
    {Cpu::ARM9, true, {0x4289, 0x4200, 0xD100, 0x3001, 0xD200, 0x3064, 0x3001}, {0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
    {Cpu::ARM7, true, {0x4289, 0x4200, 0xD100, 0x3001, 0xD200, 0x3064, 0x3001}, {5, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
    {Cpu::ARM9, true, {0x4289, 0x4200, 0xD100, 0x3001, 0x3002, 0x3064, 0x3001}, {5, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
    // ARM: the same through a conditional B, the successor reading C (addcs).
    {Cpu::ARM9, false, {0xE1510001, 0xE3100000, 0x1A000000, 0xE2900001, 0x22800064, 0xE2800001}, {5, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
    // Thumb: movs then ldr [r6 + r0] far away.
    {Cpu::ARM9, true, {0x2001, 0x5871}, {0, 0x12345678, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
    {Cpu::ARM7, true, {0x2001, 0x5871}, {0, 0x12345678, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
    // ARM7 rotates an odd-address LDRH result by 8; ARM9 does not. STRH seeds
    // unequal bytes so the rotation is visible in the final register state.
    {Cpu::ARM7, false, {0xE1C020B0, 0xE2800001, 0xE1D010B0}, {BUF_BASE, 0, 0x1234, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
    {Cpu::ARM9, false, {0xE1C020B0, 0xE2800001, 0xE1D010B0}, {BUF_BASE, 0, 0x1234, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
    // ARM7 odd-address LDRSH returns the signed low byte, unlike LDRH rotation.
    {Cpu::ARM7, false, {0xE1C020B0, 0xE2800001, 0xE1D010F0}, {BUF_BASE, 0, 0x1280, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},

    // `LDM ^` / `STM ^` -- the user-bank transfer. The interpreter switches to
    // user mode and back; the JIT inlines it in IRQ/SVC/ABT/UND, where only
    // r13/r14 are banked away from the live set, and keeps the interpreter in
    // FIQ (r8-r12 banked too) and USR/SYS (the live bank already IS bank 0).
    // Each case ends `msr cpsr_c, #SYS` so the user bank is back in hot.regs
    // for the comparison -- without it a wrong bank would be invisible.
    //
    // LDM: give the entered mode its own r13/r14 (0xAA/0xBB) first; those must
    // survive, and the loaded words must land in the user bank.
    {Cpu::ARM9, false, {0xE321F0D2, 0xE3A0D0AA, 0xE3A0E0BB, 0xE8D07FFF, 0xE321F0DF}, {BUF_BASE, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},   // IRQ
    {Cpu::ARM7, false, {0xE321F0D2, 0xE3A0D0AA, 0xE3A0E0BB, 0xE8D07FFF, 0xE321F0DF}, {BUF_BASE, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},   // IRQ
    {Cpu::ARM9, false, {0xE321F0D3, 0xE3A0D0AA, 0xE3A0E0BB, 0xE8D07FFF, 0xE321F0DF}, {BUF_BASE, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},   // SVC
    {Cpu::ARM7, false, {0xE321F0D3, 0xE3A0D0AA, 0xE3A0E0BB, 0xE8D07FFF, 0xE321F0DF}, {BUF_BASE, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},   // SVC
    {Cpu::ARM9, false, {0xE321F0D7, 0xE3A0D0AA, 0xE3A0E0BB, 0xE8D07FFF, 0xE321F0DF}, {BUF_BASE, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},   // ABT
    {Cpu::ARM7, false, {0xE321F0D7, 0xE3A0D0AA, 0xE3A0E0BB, 0xE8D07FFF, 0xE321F0DF}, {BUF_BASE, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},   // ABT
    {Cpu::ARM9, false, {0xE321F0DB, 0xE3A0D0AA, 0xE3A0E0BB, 0xE8D07FFF, 0xE321F0DF}, {BUF_BASE, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},   // UND
    {Cpu::ARM7, false, {0xE321F0DB, 0xE3A0D0AA, 0xE3A0E0BB, 0xE8D07FFF, 0xE321F0DF}, {BUF_BASE, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},   // UND
    {Cpu::ARM9, false, {0xE321F0D1, 0xE3A0D0AA, 0xE3A0E0BB, 0xE8D07FFF, 0xE321F0DF}, {BUF_BASE, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},   // FIQ
    {Cpu::ARM7, false, {0xE321F0D1, 0xE3A0D0AA, 0xE3A0E0BB, 0xE8D07FFF, 0xE321F0DF}, {BUF_BASE, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},   // FIQ
    {Cpu::ARM9, false, {0xE321F0DF, 0xE3A0D0AA, 0xE3A0E0BB, 0xE8D07FFF, 0xE321F0DF}, {BUF_BASE, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},   // SYS
    {Cpu::ARM7, false, {0xE321F0DF, 0xE3A0D0AA, 0xE3A0E0BB, 0xE8D07FFF, 0xE321F0DF}, {BUF_BASE, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},   // SYS
    // STM: the user r13/r14 (0xAA/0xBB) are set before the mode change and the
    // entered mode's are 0x11/0x22 -- the store must write the former.
    {Cpu::ARM9, false, {0xE3A0D0AA, 0xE3A0E0BB, 0xE321F0D2, 0xE3A0D011, 0xE3A0E022, 0xE8C07FFF, 0xE321F0DF}, {BUF_BASE, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},   // IRQ
    {Cpu::ARM7, false, {0xE3A0D0AA, 0xE3A0E0BB, 0xE321F0D2, 0xE3A0D011, 0xE3A0E022, 0xE8C07FFF, 0xE321F0DF}, {BUF_BASE, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},   // IRQ
    {Cpu::ARM9, false, {0xE3A0D0AA, 0xE3A0E0BB, 0xE321F0D3, 0xE3A0D011, 0xE3A0E022, 0xE8C07FFF, 0xE321F0DF}, {BUF_BASE, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},   // SVC
    {Cpu::ARM7, false, {0xE3A0D0AA, 0xE3A0E0BB, 0xE321F0D3, 0xE3A0D011, 0xE3A0E022, 0xE8C07FFF, 0xE321F0DF}, {BUF_BASE, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},   // SVC
    {Cpu::ARM9, false, {0xE3A0D0AA, 0xE3A0E0BB, 0xE321F0D7, 0xE3A0D011, 0xE3A0E022, 0xE8C07FFF, 0xE321F0DF}, {BUF_BASE, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},   // ABT
    {Cpu::ARM7, false, {0xE3A0D0AA, 0xE3A0E0BB, 0xE321F0D7, 0xE3A0D011, 0xE3A0E022, 0xE8C07FFF, 0xE321F0DF}, {BUF_BASE, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},   // ABT
    {Cpu::ARM9, false, {0xE3A0D0AA, 0xE3A0E0BB, 0xE321F0DB, 0xE3A0D011, 0xE3A0E022, 0xE8C07FFF, 0xE321F0DF}, {BUF_BASE, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},   // UND
    {Cpu::ARM7, false, {0xE3A0D0AA, 0xE3A0E0BB, 0xE321F0DB, 0xE3A0D011, 0xE3A0E022, 0xE8C07FFF, 0xE321F0DF}, {BUF_BASE, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},   // UND
    {Cpu::ARM9, false, {0xE3A0D0AA, 0xE3A0E0BB, 0xE321F0D1, 0xE3A0D011, 0xE3A0E022, 0xE8C07FFF, 0xE321F0DF}, {BUF_BASE, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},   // FIQ
    {Cpu::ARM7, false, {0xE3A0D0AA, 0xE3A0E0BB, 0xE321F0D1, 0xE3A0D011, 0xE3A0E022, 0xE8C07FFF, 0xE321F0DF}, {BUF_BASE, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},   // FIQ
    {Cpu::ARM9, false, {0xE3A0D0AA, 0xE3A0E0BB, 0xE321F0DF, 0xE3A0D011, 0xE3A0E022, 0xE8C07FFF, 0xE321F0DF}, {BUF_BASE, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},   // SYS
    {Cpu::ARM7, false, {0xE3A0D0AA, 0xE3A0E0BB, 0xE321F0DF, 0xE3A0D011, 0xE3A0E022, 0xE8C07FFF, 0xE321F0DF}, {BUF_BASE, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},   // SYS
    // Base in the list (Golden Sun's `ldm r0, {r0-r14}^`) is above; here the
    // base is outside it, the low-register-only list that needs no bank slot
    // at all, and a store with r15 in the list.
    {Cpu::ARM9, false, {0xE321F0D2, 0xE8D07FFE, 0xE321F0DF}, {BUF_BASE, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
    {Cpu::ARM9, false, {0xE321F0D2, 0xE8D000FE, 0xE321F0DF}, {BUF_BASE, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
    {Cpu::ARM9, false, {0xE321F0D2, 0xE8C0FFFF, 0xE321F0DF}, {BUF_BASE, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
    {Cpu::ARM7, false, {0xE321F0D2, 0xE8C0FFFF, 0xE321F0DF}, {BUF_BASE, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
    // A base 0x7D0 into the buffer: the 15/16 words straddle the 2 KB page
    // boundary, so the transfer takes the per-word slow path -- which has to
    // redirect r13/r14 to the user bank just as the inline one does.
    {Cpu::ARM9, false, {0xE321F0D2, 0xE3A0D0AA, 0xE3A0E0BB, 0xE8D07FFF, 0xE321F0DF}, {BUF_BASE + 0x7D0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
    {Cpu::ARM7, false, {0xE321F0D2, 0xE3A0D0AA, 0xE3A0E0BB, 0xE8D07FFF, 0xE321F0DF}, {BUF_BASE + 0x7D0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
    {Cpu::ARM9, false, {0xE3A0D0AA, 0xE3A0E0BB, 0xE321F0D2, 0xE3A0D011, 0xE3A0E022, 0xE8C07FFF, 0xE321F0DF}, {BUF_BASE + 0x7D0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
    {Cpu::ARM7, false, {0xE3A0D0AA, 0xE3A0E0BB, 0xE321F0D2, 0xE3A0D011, 0xE3A0E022, 0xE8C07FFF, 0xE321F0DF}, {BUF_BASE + 0x7D0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
    // STM with r15 in the list, based 0x7D0 into the buffer so it straddles
    // the 2 KB page and every word takes the per-word slow path. That path
    // had no r15 case and stored r14 instead (host_reg(15) IS r14's host
    // register); plain and user-bank forms both reach it.
    {Cpu::ARM9, false, {0xE880FFFF}, {BUF_BASE + 0x7D0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
    {Cpu::ARM7, false, {0xE880FFFF}, {BUF_BASE + 0x7D0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
    {Cpu::ARM9, false, {0xE321F0D2, 0xE8C0FFFF, 0xE321F0DF}, {BUF_BASE + 0x7D0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
    // `LDM ^` writeback with r13 as the base must update USER r13 while leaving
    // the exception mode's active r13 unchanged.
    {Cpu::ARM9, false, {0xE321F0D2, 0xE1A0D000, 0xE8FD00FF, 0xE321F0DF}, {BUF_BASE, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
    {Cpu::ARM7, false, {0xE321F0D2, 0xE1A0D000, 0xE8FD00FF, 0xE321F0DF}, {BUF_BASE, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
    {Cpu::ARM9, false, {0xE321F0D3, 0xE1A0E000, 0xE8FE00FF, 0xE321F0DF}, {BUF_BASE, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
    // The STM counterpart writes back to the active exception-mode r13 after
    // returning from the user-bank transfer.
    {Cpu::ARM9, false, {0xE321F0D2, 0xE1A0D000, 0xE8ED00FF, 0xE321F0DF}, {BUF_BASE, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
    // Conditional, taken and not taken (the condition wraps the whole body,
    // mode guard included).
    {Cpu::ARM9, false, {0xE321F0D2, 0xE1500000, 0x08D07FFF, 0xE321F0DF}, {BUF_BASE, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
    {Cpu::ARM9, false, {0xE321F0D2, 0xE1500000, 0x18D07FFF, 0xE321F0DF}, {BUF_BASE, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
    // Loads to pc, inline: a literal (ldr pc, [pc, #-4] with the target as the
    // next word), a post-indexed pop from the stack (push r1 first), and the
    // conditional pop the IRQ handlers end with, taken and not taken.
    {Cpu::ARM9, false, {0xE51FF004, HALT_STUB}, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
    {Cpu::ARM7, false, {0xE51FF004, HALT_STUB}, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
    {Cpu::ARM9, false, {0xE52D1004, 0xE49DF004}, {0, HALT_STUB, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
    {Cpu::ARM7, false, {0xE52D1004, 0xE49DF004}, {0, HALT_STUB, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
    {Cpu::ARM9, false, {0xE52D1004, 0xE3500000, 0x049DF004, 0xE2800001}, {0, HALT_STUB, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
    {Cpu::ARM9, false, {0xE52D1004, 0xE3500000, 0x049DF004, 0xE2800001}, {1, HALT_STUB, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
    {Cpu::ARM7, false, {0xE52D1004, 0xE3500000, 0x049DF004, 0xE2800001}, {0, HALT_STUB, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
    // SWI: the harness vector is itself `ldr pc, [pc, #k]`.
    {Cpu::ARM9, false, {0xEF000000}, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
    {Cpu::ARM7, false, {0xEF000000}, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
    // MRC p15 reads (ARM9): DTCM region, control, main ID, then an ALU op.
    {Cpu::ARM9, false, {0xEE190F11, 0xEE111F10, 0xEE102F10, 0xE2800001}, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
    // Cond=NV remains the unconditional/undefined space, not an ignored MCR.
    {Cpu::ARM9, false, {0xFE070F9A}, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
    // Unconditional PLD is a timing-only hint on ARM9 and undefined on ARM7.
    {Cpu::ARM9, false, {0xF5D0F000, 0xE2800001}, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
    {Cpu::ARM7, false, {0xF5D0F000, 0xE2800001}, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
    // MOVS pc, lr (register form) from SVC: SPSR = SYS, lr = the halt stub.
    {Cpu::ARM9, false, {0xE321F0D3, 0xE16FF001, 0xE1A0E002, 0xE1B0F00E}, {0, 0x1F, HALT_STUB, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
    {Cpu::ARM7, false, {0xE321F0D3, 0xE16FF001, 0xE1A0E002, 0xE1B0F00E}, {0, 0x1F, HALT_STUB, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
    // SMLALBB / SMLALTT r4 (lo), r5 (hi), r1, r2: a carry out of the low word, and a negative product.
    {Cpu::ARM9, false, {0xE1454281, 0xE2800001}, {0, 0x7FFF8000u, 0x00028001u, 0, 0xFFFFFFF0u, 0x10, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
    {Cpu::ARM9, false, {0xE14542E1, 0xE2800001}, {0, 0x7FFF8000u, 0x80018001u, 0, 0x10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
    // ARM9 CLZ lowers directly to MIPS32r2 SPECIAL2; test zero and boundary values.
    {Cpu::ARM9, false, {0xE16F0F11}, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
    {Cpu::ARM9, false, {0xE16F0F11}, {0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
    {Cpu::ARM9, false, {0xE16F0F11}, {0, 0x80000000u, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
    // Multi-register Thumb PUSH/POP use one mapped-RAM block transfer.
    {Cpu::ARM9, true, {0xB4F0, 0x2400, 0x2500, 0x2600, 0x2700, 0xBCF0}, {0, 0, 0, 0, 4, 5, 6, 7}},
    {Cpu::ARM7, true, {0xB4F0, 0x2400, 0x2500, 0x2600, 0x2700, 0xBCF0}, {0, 0, 0, 0, 4, 5, 6, 7}},
    {Cpu::ARM9, true, {0xB503, 0x2000, 0x2100, 0xBC03, 0xB001}, {0x1234, 0x5678}},
    {Cpu::ARM7, true, {0xB503, 0x2000, 0x2100, 0xBC03, 0xB001}, {0x1234, 0x5678}},
    // ARMv5 saturating arithmetic, including both saturation stages of QDADD/QDSUB.
    {Cpu::ARM9, false, {0xE1020051, 0xE2800003}, {0, 5, 7}},
    {Cpu::ARM9, false, {0xE1020051, 0xE2800003}, {0, 0x7fffffffu, 1}},
    {Cpu::ARM9, false, {0xE1220051, 0xE2800003}, {0, 0x80000000u, 1}},
    {Cpu::ARM9, false, {0xE1220051, 0xE2800003}, {0, 0x7fffffffu, 0xffffffffu}},
    {Cpu::ARM9, false, {0xE1420051, 0xE2800003}, {0, 7, 5}},
    {Cpu::ARM9, false, {0xE1420051, 0xE2800003}, {0, 1, 0x40000000u}},
    {Cpu::ARM9, false, {0xE1620051, 0xE2800003}, {0, 0x80000000u, 0x40000000u}},
    {Cpu::ARM9, false, {0xE1620051, 0xE2800003}, {0, 0x7fffffffu, 0xffffffffu}},
    // ARM9 LDRD/STRD, including pre/post-indexed, negative, and register offsets.
    {Cpu::ARM9, false, {0xE1C300D8, 0xE5860000, 0xE5861004}, {0, 0, 0, BUF_BASE + 0x100}},
    {Cpu::ARM9, false, {0xE1E300D8, 0xE5860000, 0xE5861004, 0xE5863008}, {0, 0, 0, BUF_BASE + 0x100}},
    {Cpu::ARM9, false, {0xE0C300D8, 0xE5863008}, {0, 0, 0, BUF_BASE + 0x100}},
    {Cpu::ARM9, false, {0xE14300D8, 0xE5860000, 0xE5861004}, {0, 0, 0, BUF_BASE + 0x100}},
    {Cpu::ARM9, false, {0xE1C300F8}, {0x11223344u, 0x55667788u, 0, BUF_BASE + 0x100}},
    {Cpu::ARM9, false, {0xE18300D4, 0xE5860000, 0xE5861004}, {0, 0, 0, BUF_BASE + 0x100, 12}},
    {Cpu::ARM9, false, {0xE0E300D4, 0xE5863008}, {0, 0, 0, BUF_BASE + 0x7fc}},
    {Cpu::ARM9, false, {0xE1090091}, {0, 0x11223344u}},
    {Cpu::ARM9, false, {0xE1491091}, {0, 0x55667788u}},
    {Cpu::ARM9, false, {0xE1030091}, {0, 0x89abcdefu, 0, BUF_BASE + 1}},
    {Cpu::ARM9, false, {0x11090091}, {0, 0x11223344u}}, // SWPNE, taken
    {Cpu::ARM9, false, {0x01090091}, {0, 0x11223344u}}, // SWPEQ, not taken
  };
  u32 n = 0, done = 0;
  for (const Directed& d : cases) {
    if (n++ < start) continue;
    if (done == limit) break;
    const bool a9 = d.which == Cpu::ARM9;
    Machine mi(d.which), mj(d.which);
    CHECK(jit::attach(mj.nds, a9, !a9));
    Trial t{};
    t.thumb = false;
    t.cpsr = 0;
    t.thumb = d.thumb;
    t.code = d.code;
    for (int i = 0; i < 15; ++i) t.regs[i] = d.regs[i];
    t.code_base = d.code_base;
    t.budget = d.budget;
    t.code_in_r9 = d.code_in_r9;
    t.require_halt = d.require_halt;
    t.strict_budget = d.strict_budget;
    t.regs[9] = BUF_BASE;
    t.regs[6] = BUF_BASE;
    t.regs[13] = STACK;
    t.cpsr = 0;
    const u64 fallback_before = jit::stats().instrs_fallback;
    const u64 translated_before = jit::stats().instrs_translated;
    const bool qdsp = !d.thumb && a9 && !d.code.empty() &&
        ds::arm::decode_arm(d.code[0]) >= AOp::QAdd &&
        ds::arm::decode_arm(d.code[0]) <= AOp::QDSub;
    const bool pld = !d.thumb && a9 && !d.code.empty() &&
        (d.code[0] >> 28) == 15 && ((d.code[0] >> 24) & 0xf7u) == 0xf5u;
    const bool dual = !d.thumb && a9 && !d.code.empty() &&
        (ds::arm::decode_arm(d.code[0]) == AOp::LdrStrHImm ||
         ds::arm::decode_arm(d.code[0]) == AOp::LdrStrHReg) &&
        !(d.code[0] & (1u << 20)) && ((d.code[0] >> 5) & 3u) >= 2;
    const bool ok = run_both(mi, mj, t, a9, 100000 + n, true);
    if (!d.thumb && a9 && !d.code.empty() &&
        (ds::arm::decode_arm(d.code[0]) == AOp::Swp ||
         ds::arm::decode_arm(d.code[0]) == AOp::Swpb))
      CHECK(jit::stats().instrs_translated > translated_before);
    if (qdsp || pld || (dual && d.regs[3] != BUF_BASE + 0x7fc))
      CHECK(jit::stats().instrs_fallback == fallback_before);
    jit::detach(mj.nds);
    if (!ok) std::fprintf(stderr, "directed case %u failed\n", n);
    CHECK(ok);
    ++done;
  }
  std::printf("jit directed: %u cases ok\n", done);
}

u32 alu_imm(u32 op, u32 rd, u32 rn, u32 rotate, u32 imm8, u32 cond = 14, bool setflags = false) {
  return (cond << 28) | (1u << 25) | (op << 21) | (setflags << 20) |
         (rn << 16) | (rd << 12) | (rotate << 8) | imm8;
}

u32 alu_reg(u32 op, u32 rd, u32 rn, u32 rm, u32 shift_type = 0, u32 shift = 0, u32 cond = 14) {
  return (cond << 28) | (op << 21) | (rn << 16) | (rd << 12) |
         (shift << 7) | (shift_type << 5) | rm;
}

u32 alu_reg_shift(u32 op, u32 rd, u32 rn, u32 rm, u32 rs, u32 shift_type) {
  return (14u << 28) | (op << 21) | (rn << 16) | (rd << 12) |
         (rs << 8) | (shift_type << 5) | 0x10u | rm;
}

#if defined(__mips__)
void native_compare_loop(Cpu which);
void native_thumb_smoke(Cpu which);
#endif

void native_alu(int only = -1) {
  // Unconditional, non-S, non-PC ARM ALU forms: each immediate rotate edge
  // and each native register form gets exercised without relying on flags.
  const u32 ops[] = {13, 15, 4, 2, 0, 1, 12, 14, 3, 7}; // common ALU plus RSB/RSC
  const u32 rotates[] = {0, 1, 4, 8, 12, 15};
  // MIPS native lowering is opt-in in production.  Make this test exercise
  // the smallest risky form deterministically instead of depending on the
  // caller's environment (or silently testing fallback code).
  setenv("DS_MIPS_NATIVE", "1", 1);
  setenv("DS_MIPS_NATIVE_ARM7", "1", 1);
  unsetenv("DS_MIPS_NATIVE_LIMIT");
#if defined(__mips__)
  const char *native_ops[] = {"0", "1", "2", "3", "4", "7", "8", "9", "12", "13", "all"};
#else
  const char *native_ops[] = {"0"};
#endif
  for (const char *native_op : native_ops) {
    setenv("DS_MIPS_NATIVE_OP", native_op, 1);
    for (Cpu which : {Cpu::ARM9, Cpu::ARM7}) {
      if (only >= 0 && static_cast<int>(which) != only) continue;
      Machine mi(which), mj(which);
      CHECK(jit::attach(mj.nds, which == Cpu::ARM9, which == Cpu::ARM7));
      Trial t{};
      t.thumb = false;
      t.cpsr = 0;
      t.code.reserve(64);
      for (u32 i = 0; i < sizeof ops / sizeof ops[0]; ++i) {
        const u32 op = ops[i];
        t.code.push_back(alu_imm(op, i & 7, (i + 1) & 7, rotates[i % 6], 0x11u + i * 0x13u));
        t.code.push_back(alu_reg(op, (i + 2) & 7, i & 7, (i + 3) & 7));
      }
      for (u32 i = 0; i < 24; ++i) {
        const u32 op = ops[i & 7];
        t.code.push_back(alu_imm(op, (i + 1) & 7, i & 7, rotates[(i + 2) % 6], 0xA5u ^ i));
      }
      for (u32 shift : {1u, 7u, 15u, 31u})
        t.code.push_back(alu_reg(0, 0, 1, 2, 3, shift)); // ROR (not RRX)
      t.code.push_back(alu_reg(0, 0, 1, 2) | (1u << 20)); // ANDS, preserve C/V
      t.code.push_back(alu_reg(1, 2, 3, 4) | (1u << 20)); // EORS
      t.code.push_back(alu_reg(12, 3, 4, 5) | (1u << 20)); // ORRS
      t.code.push_back(alu_reg(13, 4, 0, 5) | (1u << 20)); // MOVS
      t.code.push_back(alu_imm(0, 0, 1, 4, 0x81, 14, true)); // rotated immediate sets C
      t.code.push_back(alu_imm(12, 1, 2, 0, 0, 14, true)); // zero immediate preserves C
      for (int i = 0; i < 15; ++i) t.regs[i] = 0x10203040u + static_cast<u32>(i) * 0x11111111u;
      t.regs[9] = BUF_BASE;
      t.regs[13] = STACK;
      t.require_halt = true;
      CHECK(run_both(mi, mj, t, which == Cpu::ARM9, which == Cpu::ARM9 ? 91001 : 71001, true));
      if (std::strcmp(native_op, "all") == 0) {
        Trial pc_operand{};
        pc_operand.require_halt = true;
        pc_operand.code = {alu_imm(4, 0, 15, 0, 8),
                           alu_imm(2, 1, 15, 0, 12),
                           alu_imm(10, 0, 15, 0, 0) | (1u << 20),
                           alu_reg(13, 2, 0, 15), // MOV r2,pc
                           alu_reg(1, 3, 1, 15, 2, 5)};
        pc_operand.regs[13] = STACK;
        CHECK(run_both(mi, mj, pc_operand, which == Cpu::ARM9,
                       which == Cpu::ARM9 ? 100900 : 80900, true));
        Trial msr_flags{};
        msr_flags.require_halt = true;
        msr_flags.cpsr = 0x0000001fu;
        msr_flags.code = {0xE128F003u, 0xE10F4000u, 0xE328F20Fu,
                          0xE10F5000u, 0xE5844000u, 0xE5845004u};
        msr_flags.regs[3] = 0xa1234567u;
        msr_flags.regs[4] = BUF_BASE;
        msr_flags.regs[13] = STACK;
        CHECK(run_both(mi, mj, msr_flags, which == Cpu::ARM9,
                       which == Cpu::ARM9 ? 100950 : 80950, true));
        for (u32 carry : {0u, 1u}) {
          Trial rrx{};
          rrx.require_halt = true;
          rrx.cpsr = carry << 29;
          rrx.code = {alu_reg(13, 0, 0, 1, 3, 0) | (1u << 20),
                      0xE10F2000u, 0xE5860000u, 0xE5862004u};
          rrx.regs[1] = 0x80000003u;
          rrx.regs[6] = BUF_BASE;
          rrx.regs[13] = STACK;
          CHECK(run_both(mi, mj, rrx, which == Cpu::ARM9,
                         (which == Cpu::ARM9 ? 101000 : 81000) + carry, true));
        }
        if (which == Cpu::ARM7) {
          Trial skipped_mul{};
          skipped_mul.cpsr = 0x40000000u; // Z=1 makes this NE multiply skip.
          skipped_mul.code = {0x102a289c};
          skipped_mul.regs[8] = 0xb825fc23u;
          skipped_mul.regs[12] = 0x249e1b58u;
          skipped_mul.regs[13] = STACK;
          skipped_mul.require_halt = true;
          CHECK(run_both(mi, mj, skipped_mul, false, 71020, true));
        }
        Trial thumb_lsl{};
        thumb_lsl.thumb = true;
        thumb_lsl.code = {0x006b}; // LSL r3, r5, #1: carry must come from bit 31.
        thumb_lsl.regs[5] = 0x80000001u;
        thumb_lsl.regs[13] = STACK;
        thumb_lsl.require_halt = true;
        CHECK(run_both(mi, mj, thumb_lsl, which == Cpu::ARM9,
                       (which == Cpu::ARM9 ? 101100 : 81100), true));
        static constexpr u32 mul_rhs[] = {
            0u, 0x7fu, 0xffffff80u, 0x100u, 0xffffu,
            0x10000u, 0x01000000u, 0x7fffffffu, 0x80000000u, 0xffffffffu};
        for (u32 form = 0; form < 4; ++form) {
          for (u32 i = 0; i < sizeof mul_rhs / sizeof mul_rhs[0]; ++i) {
            const u32 rhs = mul_rhs[i];
            Trial mul{};
            mul.require_halt = true;
            const u32 op = form & 1 ? 0xE0235493u : 0xE0020493u;
            mul.code = {op | ((form & 2) ? (1u << 20) : 0),
                        0xE10F6000u, 0xE5812000u, 0xE5816004u};
            mul.cpsr = 0xf0000000u;
            mul.regs[1] = BUF_BASE;
            mul.regs[3] = 0x12345678u;
            mul.regs[4] = rhs;
            mul.regs[5] = 0x87654321u;
            mul.regs[13] = STACK;
            CHECK(run_both(mi, mj, mul, which == Cpu::ARM9,
                           (which == Cpu::ARM9 ? 102000 : 82000) + form * 16 + i,
                           true));
          }
        }
        for (u32 shift_type = 0; shift_type < 4; ++shift_type) {
          for (u32 amount : {0u, 1u, 31u, 32u, 33u, 255u}) {
            Trial variable{};
            variable.require_halt = true;
            variable.code = {
                alu_reg_shift(13, 0, 0, 2, 3, shift_type), // MOV r0,r2,shift r3
                alu_reg_shift(4, 4, 1, 2, 3, shift_type),  // ADD r4,r1,r2,shift r3
                0xE5860000u, 0xE5864004u};
            variable.regs[1] = 0x12345678u;
            variable.regs[2] = 0x80000001u;
            variable.regs[3] = amount;
            variable.regs[6] = BUF_BASE;
            variable.regs[13] = STACK;
            CHECK(run_both(mi, mj, variable, which == Cpu::ARM9,
                           (which == Cpu::ARM9 ? 96000 : 76000) + shift_type * 512 + amount,
                           true));
          }
        }
        for (u32 op : {2u, 3u, 4u, 5u, 6u, 7u, 10u, 11u}) {
          for (u32 shift_type = 0; shift_type < 4; ++shift_type) {
            for (u32 amount : {0u, 1u, 32u, 33u, 255u}) {
              Trial variable_flags{};
              variable_flags.require_halt = true;
              variable_flags.cpsr = ((op + shift_type + amount) & 15u) << 28;
              variable_flags.code = {
                  alu_reg_shift(op, 0, 1, 2, 3, shift_type) | (1u << 20),
                  0xE10F4000u, 0xE5864000u}; // MRS r4,CPSR; STR r4,[r6]
              variable_flags.regs[1] = 0x7fffffffu;
              variable_flags.regs[2] = 0x80000001u;
              variable_flags.regs[3] = amount;
              variable_flags.regs[6] = BUF_BASE;
              variable_flags.regs[13] = STACK;
              CHECK(run_both(mi, mj, variable_flags, which == Cpu::ARM9,
                             (which == Cpu::ARM9 ? 98000 : 78000) + op * 256 +
                                 shift_type * 64 + amount,
                             true));
            }
          }
        }
        for (u32 op : {0u, 1u, 8u, 9u, 12u, 13u, 14u, 15u}) {
          for (u32 shift_type = 0; shift_type < 4; ++shift_type) {
            for (u32 amount : {0u, 1u, 31u, 32u, 33u, 255u}) {
              Trial variable_flags{};
              variable_flags.require_halt = true;
              variable_flags.cpsr = ((op + shift_type + amount) & 15u) << 28;
              variable_flags.code = {
                  alu_reg_shift(op, 0, 1, 2, 3, shift_type) | (1u << 20),
                  0xE10F4000u, 0xE5864000u}; // MRS r4,CPSR; STR r4,[r6]
              variable_flags.regs[1] = 0x7fffffffu;
              variable_flags.regs[2] = 0x80000001u;
              variable_flags.regs[3] = amount;
              variable_flags.regs[6] = BUF_BASE;
              variable_flags.regs[13] = STACK;
              CHECK(run_both(mi, mj, variable_flags, which == Cpu::ARM9,
                             (which == Cpu::ARM9 ? 100000 : 80000) + op * 256 +
                                 shift_type * 64 + amount,
                             true));
            }
          }
        }
      }
      if (std::strcmp(native_op, "all") == 0) {
        for (u32 flags = 0; flags < 16; ++flags) {
          Trial pred{};
          pred.require_halt = true;
          pred.cpsr = flags << 28;
          for (u32 cond = 0; cond < 14; ++cond)
            pred.code.push_back(alu_imm(13, cond, 0, 0, 0x80 + cond, cond));
          for (int i = 0; i < 15; ++i) pred.regs[i] = 0x10203040u + static_cast<u32>(i);
          pred.regs[9] = BUF_BASE;
          pred.regs[13] = STACK;
          CHECK(run_both(mi, mj, pred, which == Cpu::ARM9,
                         (which == Cpu::ARM9 ? 92000 : 72000) + flags, true));
        }
        for (u32 op : {0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u, 10u, 11u, 12u, 13u}) {
          for (u32 form = 0; form < 3; ++form) {
            for (u32 flags = 0; flags < 16; ++flags) {
              Trial flag_case{};
              flag_case.require_halt = true;
              flag_case.cpsr = flags << 28;
              if (form == 0)
                flag_case.code = {alu_reg(op, 0, 1, 2) | (1u << 20)};
              else
                flag_case.code = {alu_imm(op, 0, 1, form == 1 ? 4 : 8,
                                          form == 1 ? 0x81 : 0x7f, 14, true)};
              for (int i = 0; i < 15; ++i) flag_case.regs[i] = 0x10203040u + static_cast<u32>(i);
              flag_case.regs[1] = flags & 1 ? 0x80000000u : 0x7fffffffu;
              flag_case.regs[2] = flags & 2 ? 0xffffffffu : 0x80000000u;
              flag_case.regs[9] = BUF_BASE;
              flag_case.regs[13] = STACK;
              CHECK(run_both(mi, mj, flag_case, which == Cpu::ARM9,
                             (which == Cpu::ARM9 ? 93000 : 73000) + op * 48 + form * 16 + flags, true));
            }
          }
        }
        const u32 shifts[][2] = {{0, 1}, {0, 31}, {1, 0}, {2, 0}, {3, 1}, {3, 31}};
        for (u32 op : {0u, 1u, 8u, 9u, 12u, 13u}) {
          for (const auto& shift : shifts) {
            const u32 type = shift[0], amount = shift[1];
            for (u32 flags = 0; flags < 16; ++flags) {
              Trial shifted{};
              shifted.require_halt = true;
              shifted.cpsr = flags << 28;
              shifted.code = {alu_reg(op, 0, 1, 2, type, amount) | (1u << 20)};
              for (int i = 0; i < 15; ++i) shifted.regs[i] = 0x10203040u + static_cast<u32>(i);
              shifted.regs[1] = flags & 1 ? 0x80000000u : 0x7fffffffu;
              shifted.regs[2] = flags & 2 ? 0xffffffffu : 0x80000001u;
              shifted.regs[9] = BUF_BASE;
              shifted.regs[13] = STACK;
              CHECK(run_both(mi, mj, shifted, which == Cpu::ARM9,
                             93500 + op * 128 + type * 32 + amount + flags, true));
            }
          }
        }
      }
      jit::detach(mj.nds);
    }
  }
#if defined(__mips__)
  if (only != static_cast<int>(Cpu::ARM7)) {
    setenv("DS_MIPS_NATIVE_OP", "all", 1);
    native_compare_loop(Cpu::ARM9);
  }
  if (only != static_cast<int>(Cpu::ARM9)) {
    setenv("DS_MIPS_NATIVE_OP", "all", 1);
    native_compare_loop(Cpu::ARM7);
  }
  setenv("DS_MIPS_NATIVE_OP", "all", 1);
  if (only != static_cast<int>(Cpu::ARM7)) native_thumb_smoke(Cpu::ARM9);
  if (only != static_cast<int>(Cpu::ARM9)) native_thumb_smoke(Cpu::ARM7);
#endif
  unsetenv("DS_MIPS_NATIVE_OP");
  unsetenv("DS_MIPS_NATIVE");
  unsetenv("DS_MIPS_NATIVE_ARM7");
  // native_compare_loop enables these opt-in paths for its canaries; don't
  // accidentally turn the randomized baseline fuzz into another native run.
  unsetenv("DS_MIPS_NATIVE_BRANCH");
  unsetenv("DS_MIPS_NATIVE_MEM");
  unsetenv("DS_MIPS_NATIVE_MEM_ARM7");
  std::puts("jit native ALU: immediate+register forms ok");
}

#if defined(__mips__)
__attribute__((noinline)) void native_compare_loop(Cpu which) {
  static Machine mi9(Cpu::ARM9), mj9(Cpu::ARM9);
  static Machine mi7(Cpu::ARM7), mj7(Cpu::ARM7);
  Machine& mi = which == Cpu::ARM9 ? mi9 : mi7;
  Machine& mj = which == Cpu::ARM9 ? mj9 : mj7;
  const bool a9 = which == Cpu::ARM9;
  setenv("DS_MIPS_NATIVE_BRANCH", "1", 1);
  setenv("DS_MIPS_NATIVE_MEM", "1", 1);
  CHECK(jit::attach(mj.nds, a9, !a9));
  Trial t{};
  // The scheduler's clock advances inside native execution. Verify MMIO
  // stores and loads are ordered before their respective instruction charge.
  t.require_halt = true; t.regs[0] = 0x04000100; t.regs[1] = 0x80;
  t.code = {0xE1C010B2, 0xE1D020B0}; // start timer; ldrh r2, [r0]
  jit::flush(mj.cpu);
  load_trial(mi, t, a9);
  load_trial(mj, t, a9);
  for (Machine* m : {&mi, &mj}) {
    m->nds.bus.write16(which, 0x04000102, 0);
    m->nds.bus.write16(which, 0x04000100, 0x1000);
    m->nds.cpu(a9 ? Cpu::ARM7 : Cpu::ARM9).halted = true;
    m->nds.sched.run_until(m->nds.sched.now() + 200);
  }
  CHECK(compare(mi, mj, t, a9 ? 91201 : 71201, true));
  if (a9) {
    t = Trial{}; t.require_halt = true; t.regs[14] = HALT_STUB;
    t.code = {0xE1A0000E, 0xE1A0000E, 0xE1A0000E, 0xE1A0000E,
              0xE1A0000E, 0xEB000001, 0xEA000002, 0xE1A00000,
              0xE3A01077, 0xE12FFF1E};
    CHECK(run_both(mi, mj, t, true, 91020, true));
    CHECK(mj.cpu.hot.regs[1] == 0x77);
    t = Trial{}; t.thumb = true; t.require_halt = true; t.regs[14] = HALT_STUB;
    t.code = {0xF000, 0xF808, 0xE00C, 0x46C0, 0x46C0, 0x46C0, 0x46C0,
              0x46C0, 0x46C0, 0x46C0, 0x2177, 0x4770, 0x4770, 0x4770,
              0x4770, 0x4770};
    CHECK(run_both(mi, mj, t, true, 91019, true));
    CHECK(mj.cpu.hot.regs[1] == 0x77);
    struct DspCase { u32 ins, rm, rs, rn; };
    static constexpr DspCase dsp[] = {
      {0xE1600281, 0x7FFF8000u, 0x00028001u, 0}, // SMULBB
      {0xE16002E1, 0x7FFF8000u, 0x00028001u, 0}, // SMULTT
      {0xE16002C1, 0x7FFF8000u, 0x00028001u, 0}, // SMULBT
      {0xE1003281, 0x00001234u, 0x00005678u, 0x100}, // SMLABB
      {0xE1003281, 0x7FFF7FFFu, 0x7FFF7FFFu, 0x7FFFFFFFu}, // SMLABB sets sticky Q
      {0xE1203281, 0x12345678u, 0x00007FFFu, 0x100}, // SMLAWB
      {0xE12032C1, 0x80000000u, 0x8000FFFFu, 0x7FFFFFFFu}, // SMLAWT, Q overflow
      {0xE12002A1, 0x12345678u, 0x00007FFFu, 0}, // SMULWB
      {0xE12002E1, 0x80000000u, 0x8000FFFFu, 0}, // SMULWT
    };
    for (u32 i = 0; i < std::size(dsp); ++i) {
      t = Trial{}; t.require_halt = true;
      t.regs[1] = dsp[i].rm; t.regs[2] = dsp[i].rs; t.regs[3] = dsp[i].rn;
      t.regs[13] = STACK; t.code = {dsp[i].ins, 0xE2800001};
      CHECK(run_both(mi, mj, t, true, 91200 + i, true));
    }
    struct SmlalCase { u32 ins, hi, lo, rm, rs; };
    static constexpr SmlalCase smlal[] = {
      {0xE1445281, 0x00000001u, 0xfffffff0u, 0x00008000u, 0x00007fffu},
      {0xE1476283, 0x7fffffffu, 0x00000000u, 0x80000000u, 0x8000ffffu},
      {0xE1413281, 0x12345678u, 0x87654321u, 0xa5a55a5au, 0x5a5aa5a5u},
    };
    for (u32 i = 0; i < std::size(smlal); ++i) {
      t = Trial{}; t.require_halt = true;
      t.regs[(smlal[i].ins >> 16) & 15] = smlal[i].hi;
      t.regs[(smlal[i].ins >> 12) & 15] = smlal[i].lo;
      t.regs[smlal[i].ins & 15] = smlal[i].rm;
      t.regs[(smlal[i].ins >> 8) & 15] = smlal[i].rs;
      t.regs[13] = STACK; t.code = {smlal[i].ins, 0xE2800001};
      CHECK(run_both(mi, mj, t, true, 91210 + i, true));
    }
    static constexpr u32 mrc[] = {0xEE100F10, 0xEE190F11, 0x0E100F10};
    for (u32 i = 0; i < std::size(mrc); ++i) {
      t = Trial{}; t.require_halt = true; t.regs[13] = STACK;
      t.cpsr = i == 2 ? 0 : 0x40000000u;
      t.code = {mrc[i], 0xE2800001};
      CHECK(run_both(mi, mj, t, true, 91230 + i, true));
    }
    t = Trial{}; t.require_halt = true; t.regs[13] = STACK;
    t.code = {0xEE070F90}; // MCR p15,0,r0,c7,c0,4: wait for interrupt.
    CHECK(run_both(mi, mj, t, true, 91233, true));
    t = Trial{}; t.require_halt = true; t.regs[13] = STACK;
    t.code = {0x0E070F90, 0xE2800001}; // conditional WFI is skipped with Z clear.
    CHECK(run_both(mi, mj, t, true, 91234, true));
    CHECK(mj.cpu.hot.regs[0] == 1);
    t = Trial{}; t.require_halt = true; t.cpsr = 0x40000000u;
    t.regs[0] = 0x12345678u; t.regs[13] = STACK;
    t.code = {0x11003281, 0xE2800001}; // SMLABBNE skipped with Z set.
    CHECK(run_both(mi, mj, t, true, 91220, true));
  } else {
    // ARM7 STM from WRAM into WRAM: both instruction and data waits are
    // non-main, so the interpreter adds them rather than combining waits.
    t = Trial{}; t.require_halt = true; t.code_base = 0x03800000;
    t.regs[0] = 0x11223344; t.regs[1] = 0x55667788;
    t.regs[2] = 0x99aabbcc; t.regs[3] = 0xddeeff00;
    t.regs[12] = 0x12345678; t.regs[13] = 0x03804000;
    t.code = {0xE92D500F}; // stmdb sp!, {r0-r3,r12,lr}
    CHECK(run_both(mi, mj, t, false, 71200, true));
    CHECK(std::memcmp(mi.host(0x03803fe8), mj.host(0x03803fe8), 24) == 0);
  }
  if (a9) {
    t = Trial{}; t.require_halt = true; t.code_base = 0x020056b0;
    t.code = {0xE59F0014, 0xE1A00000, 0xE1A00000, 0xE1A00000,
              0xE1A00000, 0xE1A00000, 0xEA000000, 0xDEADBEEF};
    CHECK(run_both(mi, mj, t, true, 91021, true));
    CHECK(mj.cpu.hot.regs[0] == 0xDEADBEEF);
  }
  // Cover immediate/register-offset PC loads, including the ARM9 native path.
  static constexpr u32 pc_load[] = {0xE51FF004, 0xE51FF003, 0xE79FF10C};
  static constexpr u32 pc_literal[] = {HALT_STUB, (HALT_STUB << 8) | (HALT_STUB >> 24), HALT_STUB};
  static constexpr u32 pc_load_code[][4] = {
      {pc_load[0], pc_literal[0], 0xEAFFFFFE, 0},
      {pc_load[1], pc_literal[1], 0xEAFFFFFE, 0},
      {pc_load[2], 0xEAFFFFFE, 0xEAFFFFFE, pc_literal[2]},
  };
  for (u32 i = 0; i < 3; ++i) {
    t = Trial{};
    t.require_halt = true; t.regs[12] = 1; t.regs[13] = STACK;
    t.code.assign(pc_load_code[i], pc_load_code[i] + (i == 2 ? 4 : 3));
    CHECK(run_both(mi, mj, t, a9, (a9 ? 91022 : 71022) + i, true));
  }
  t = Trial{};
  t.require_halt = true; t.regs[1] = BUF_BASE; t.regs[2] = HALT_STUB;
  t.regs[13] = STACK; t.code = {0xE5812000, 0xE591F000}; // STR target; LDR pc,[r1]
  CHECK(run_both(mi, mj, t, a9, a9 ? 91024 : 71024, true));
  t = Trial{};
  t.require_halt = true; t.regs[0] = 0x12345678u; t.regs[1] = BUF_BASE;
  t.regs[3] = BUF_BASE + 0x20; t.regs[13] = STACK;
  t.code = {0xE5A10004, 0xE4912004, 0xE5831000, 0xE5832004};
  CHECK(run_both(mi, mj, t, a9, a9 ? 91028 : 71028, true));
  u32 writeback_results[2]{};
  std::memcpy(writeback_results, mj.host(BUF_BASE + 0x20), sizeof writeback_results);
  CHECK(writeback_results[0] == BUF_BASE + 8 && writeback_results[1] == 0x12345678u);
  t = Trial{};
  t.require_halt = true; t.regs[1] = BUF_BASE; t.regs[3] = BUF_BASE + 0x100;
  t.regs[13] = STACK;
  t.code = {0xE28F0004, 0xE5810004, 0xE8B18004, 0xE5831000};
  CHECK(run_both(mi, mj, t, a9, a9 ? 71142 : 71143, true));
  u32 ldm_pc_wb = 0;
  std::memcpy(&ldm_pc_wb, mj.host(BUF_BASE + 0x100), sizeof ldm_pc_wb);
  CHECK(ldm_pc_wb == BUF_BASE + 8);
  t = Trial{};
  t.require_halt = true; t.regs[0] = 0x12345678u; t.regs[1] = BUF_BASE;
  t.regs[2] = 0xa5a55a5au; t.regs[6] = BUF_BASE + 0x100; t.regs[13] = STACK;
  t.code = {0xE8810007, 0xE891001A, 0xE5863000, 0xE5864004, 0xE5861008};
  CHECK(run_both(mi, mj, t, a9, a9 ? 91146 : 71146, true));
  u32 base_overlap[3]{};
  std::memcpy(base_overlap, mj.host(BUF_BASE + 0x100), sizeof base_overlap);
  CHECK(base_overlap[0] == BUF_BASE && base_overlap[1] == 0xa5a55a5au &&
        base_overlap[2] == 0x12345678u);
  t = Trial{};
  t.require_halt = true; t.regs[0] = BUF_BASE; t.regs[1] = 0x12345000u;
  t.regs[2] = 0x87654321u; t.regs[13] = STACK; t.regs[14] = HALT_STUB;
  t.code = {0xE5801000, 0xE5802004, 0xE321F0D2, 0xE8D03000, 0xE8C03000,
            0xE321F0DF, 0xE580D008, 0xE580E00C};
  CHECK(run_both(mi, mj, t, a9, a9 ? 91147 : 71147, true));
  u32 user_bank[4]{};
  std::memcpy(user_bank, mj.host(BUF_BASE), sizeof user_bank);
  CHECK(user_bank[0] == 0x12345000u && user_bank[1] == 0x87654321u &&
        user_bank[2] == 0x87654321u && user_bank[3] == HALT_STUB);
  t = Trial{};
  t.require_halt = true; t.regs[1] = BUF_BASE; t.regs[6] = BUF_BASE + 0x100;
  t.regs[13] = STACK;
  t.code = {0xE321F0D2, 0xE8F10003, 0xE321F0DF, 0xE5860000, 0xE5861004};
  CHECK(run_both(mi, mj, t, a9, a9 ? 91148 : 71148, true));
  u32 loaded_base[2]{};
  std::memcpy(loaded_base, mj.host(BUF_BASE + 0x100), sizeof loaded_base);
  CHECK(loaded_base[0] == 0xa5a5a5a5u && loaded_base[1] == 0xa4a4a4a4u);
  t = Trial{};
  t.require_halt = true; t.regs[1] = HALT_STUB; t.regs[13] = STACK;
  t.code = {0xF12FFF11, 0xE2800001}; // NV BX is skipped, not taken unconditionally.
  CHECK(run_both(mi, mj, t, a9, a9 ? 91144 : 71144, true));
  if (!a9) {
    t = Trial{};
    t.require_halt = true; t.regs[1] = BUF_BASE; t.regs[13] = STACK;
    t.code = {0xE3A00080, 0xE5C10001, 0xE1D120F1, 0xE5812000};
    CHECK(run_both(mi, mj, t, false, 71140, true));
    u32 signed_half = 0;
    std::memcpy(&signed_half, mj.host(BUF_BASE), sizeof signed_half);
    CHECK(signed_half == 0xffffff80u);

    t = Trial{};
    t.require_halt = true; t.regs[1] = BUF_BASE; t.regs[13] = STACK;
    t.code = {0xE3A00080, 0xE5C10003, 0xE3A00001, 0xE5C10002,
              0xE1D120F2, 0xE5812000};
    CHECK(run_both(mi, mj, t, false, 71141, true));
    std::memcpy(&signed_half, mj.host(BUF_BASE), sizeof signed_half);
    CHECK(signed_half == 0xffff8001u);
  }
  for (u32 op : {2u, 4u}) for (bool flags : {false, true})
    for (u32 lhs : {0x7fffffffu, 0x80000000u})
      for (const auto& imm : {std::pair<u32, u32>{0, 1}, {0, 0xff}, {7, 2}, {8, 0xff}}) {
        t = Trial{};
        t.require_halt = true; t.regs[1] = lhs; t.regs[13] = STACK;
        t.code = {alu_imm(op, 0, 1, imm.first, imm.second, 14, flags)};
        CHECK(run_both(mi, mj, t, a9, (a9 ? 92000 : 72000) + op * 64 +
                       flags * 32 + (lhs >> 31) * 16 + imm.first * 512 + imm.second,
                       true));
      }
  for (u32 ldm : {0xE8BD8000u, 0x08BD8000u}) {
    t = Trial{};
    t.require_halt = true; t.cpsr = 0x4000001fu; t.regs[13] = STACK;
    t.code = {0xE59F0008, 0xE58D0000, ldm, 0xEAFFFFFE, HALT_STUB};
    CHECK(run_both(mi, mj, t, a9, a9 ? 91025 : 71025, true));
    CHECK(mj.cpu.hot.regs[13] == STACK + 4);
  }
  for (u32 pc_alu : {0xE1A0F004u, 0xE08FF102u}) {
    t = Trial{};
    t.require_halt = true; t.regs[2] = 0x3fe; t.regs[4] = HALT_STUB;
    t.code = {pc_alu, 0xEAFFFFFE};
    CHECK(run_both(mi, mj, t, a9, a9 ? 91027 : 71027, true));
  }
  t = Trial{};
  t.require_halt = true; t.regs[2] = 2; t.regs[5] = 3;
  t.code = {alu_reg_shift(4, 3, 15, 5, 2, 0)};
  CHECK(run_both(mi, mj, t, a9, a9 ? 91030 : 71030, true));
  CHECK(mj.cpu.hot.regs[3] == CODE_BASE + 24);
  t = Trial{};
  t.require_halt = true; t.regs[2] = 0;
  t.regs[5] = HALT_STUB - (CODE_BASE + 12);
  t.code = {alu_reg_shift(4, 15, 15, 5, 2, 0), 0xEAFFFFFE};
  CHECK(run_both(mi, mj, t, a9, a9 ? 91031 : 71031, true));
  for (u32 thumb_pc : {0x4687u, 0x4487u}) {
    t = Trial{};
    t.thumb = true; t.require_halt = true;
    t.regs[0] = thumb_pc == 0x4687u ? CODE_BASE + 2 : 0xfffffffeu;
    t.code = {thumb_pc, 0xDF00, 0x46C0};
    CHECK(run_both(mi, mj, t, a9, a9 ? 91028 : 71028, true));
  }
  t = Trial{};
  t.thumb = true; t.require_halt = true;
  t.regs[14] = HALT_STUB; t.code = {0x4770}; // BX lr.
  CHECK(run_both(mi, mj, t, a9, a9 ? 91032 : 71032, true));
  for (bool signed_mul : {false, true}) for (bool accumulate : {false, true})
    for (bool setflags : {false, true})
      for (u32 rhs : {0u, 0x7fu, 0x8000u, 0x800000u,
                      0x80000000u, 0xffffff80u}) {
        t = Trial{};
        t.require_halt = true;
        const u32 instr = 0xE0800090u | (signed_mul ? 0 : 1u << 22) |
            (accumulate ? 1u << 21 : 0) | (setflags ? 1u << 20 : 0) |
            (6u << 16) | (7u << 12) | (4u << 8) | 3u;
        t.code = {instr, 0xE5816000u, 0xE5817004u,
                  0xE10F5000u, 0xE5815008u};
        t.cpsr = 0xf0000000u;
        t.regs[1] = BUF_BASE; t.regs[3] = 0x87654321u; t.regs[4] = rhs;
        t.regs[6] = 0x12345678u; t.regs[7] = 0xabcdef01u; t.regs[13] = STACK;
        CHECK(run_both(mi, mj, t, a9,
                       (a9 ? 91100 : 71100) + signed_mul * 64 +
                           accumulate * 32 + setflags * 16 + rhs / 0x1000000u,
                       true));
      }
  if (a9) {
    t = Trial{};
    t.thumb = true; t.require_halt = true;
    t.regs[0] = HALT_STUB; t.code = {0x4780}; // BLX r0.
    CHECK(run_both(mi, mj, t, true, 91033, true));
    CHECK(mj.cpu.hot.regs[14] == CODE_BASE + 3);

    t = Trial{};
    t.require_halt = true; t.regs[14] = HALT_STUB; t.regs[13] = STACK;
    t.code = {0xE12FFF3E}; // BLX lr must branch using old lr, then replace it.
    CHECK(run_both(mi, mj, t, true, 91034, true));
    CHECK(mj.cpu.hot.regs[14] == CODE_BASE + 4);

    t = Trial{};
    t.require_halt = true; t.cpsr = 0; t.regs[1] = HALT_STUB;
    t.regs[14] = 0x12345678; t.regs[13] = STACK;
    t.code = {0x012FFF31, 0xE2800001}; // EQ BLX r1 is not taken; LR is unchanged.
    CHECK(run_both(mi, mj, t, true, 91035, true));
    CHECK(mj.cpu.hot.regs[14] == 0x12345678);
  }
  if (std::getenv("DS_MIPS_PC_ONLY")) {
    jit::detach(mj.nds);
    return;
  }

  t.require_halt = true;
  t.code = {alu_reg(10, 0, 0, 1) | (1u << 20),
            alu_imm(4, 0, 0, 0, 1, 3), 0x3AFFFFFC,
            0xE1A02000}; // CMP; ADDCC; BCC loop; MOV r2,r0
  for (int i = 0; i < 15; ++i) t.regs[i] = 0;
  t.regs[1] = 5;
  t.regs[9] = BUF_BASE;
  t.regs[13] = STACK;
  CHECK(run_both(mi, mj, t, a9, a9 ? 91002 : 71002, true));
  CHECK(mj.cpu.hot.regs[2] == 5);

  t = Trial{};
  t.require_halt = true;
  t.code = {0xEA000002, 0xE3A00001, 0xE5890004, 0xE3A00002,
            0xE3A00003, 0xE5890000}; // B skips a visible store
  for (int i = 0; i < 15; ++i) t.regs[i] = 0;
  t.regs[9] = BUF_BASE; t.regs[13] = STACK;
  CHECK(run_both(mi, mj, t, a9, a9 ? 91005 : 71005, true));
  u32 result = 0;
  std::memcpy(&result, mj.host(BUF_BASE), sizeof result);
  CHECK(result == 3);
  std::memcpy(&result, mj.host(BUF_BASE + 4), sizeof result);
  CHECK(result == 0xA4A4A4A4u);

  t = Trial{};
  t.require_halt = true;
  t.code = {0xE129F001, 0xE129F002}; // Same-mode CPSR control writes stay native.
  for (int i = 0; i < 15; ++i) t.regs[i] = 0;
  t.regs[1] = 0x5f; t.regs[2] = 0x1f; t.regs[13] = STACK;
  CHECK(run_both(mi, mj, t, a9, a9 ? 91013 : 71013, true));

  t = Trial{};
  t.require_halt = true;
  t.code = {0xE129F001, 0xE1A0D004, 0xE129F002,
            0xE129F001, 0xE1A0500D, 0xE129F002}; // Switch banks, write IRQ SP, then read it back.
  for (int i = 0; i < 15; ++i) t.regs[i] = 0;
  t.regs[1] = 0x12; t.regs[2] = 0x1f; t.regs[4] = 0x12345678;
  t.regs[13] = STACK;
  CHECK(run_both(mi, mj, t, a9, a9 ? 91014 : 71014, true));
  CHECK(mi.cpu.hot.regs[5] == 0x12345678 && mj.cpu.hot.regs[5] == 0x12345678);

  t = Trial{};
  t.require_halt = true;
  t.code = {0x0129F001}; // MSREQ control write skipped when Z is clear.
  for (int i = 0; i < 15; ++i) t.regs[i] = 0;
  t.regs[1] = 0x12; t.regs[13] = STACK;
  CHECK(run_both(mi, mj, t, a9, a9 ? 91015 : 71015, true));

  t = Trial{};
  t.require_halt = true;
  t.spsr = 0xD3A5F017u;
  t.code = {0xE14F0000, 0xE5810000}; // MRS r0,SPSR; STR r0,[r1]
  t.regs[1] = BUF_BASE; t.regs[13] = STACK;
  CHECK(run_both(mi, mj, t, a9, a9 ? 91016 : 71016, true));
  std::memcpy(&result, mj.host(BUF_BASE), sizeof result);
  CHECK(result == t.spsr);

  static constexpr u32 exc_return[] = {0xE1B0F00E, 0xE25EF004, 0xE29EF004};
  static constexpr u32 exc_target[] = {HALT_STUB, HALT_STUB + 4, HALT_STUB - 4};
  for (u32 i = 0; i < 3; ++i) {
    t = Trial{};
    t.require_halt = true;
    t.regs[2] = exc_target[i]; t.regs[13] = STACK;
    // A wrong exception-return PC falls into a self-loop, not the generated
    // fall-through halt branch, so this catches stale-PC restoration after
    // the helper returns.
    t.code = {0xE321F013, 0xE1A0E002, 0xE361F01F, exc_return[i], 0xEAFFFFFE};
    CHECK(run_both(mi, mj, t, a9, (a9 ? 91019 : 71019) + i, true));
  }

  t = Trial{};
  t.require_halt = true;
  t.spsr = 0x12345678u; t.regs[1] = BUF_BASE; t.regs[4] = 0xa5b6c7d8u;
  t.regs[13] = STACK;
  t.code = {0xE321F013, 0xE169F004, 0xE14F0000, 0xE5810000};
  CHECK(run_both(mi, mj, t, a9, a9 ? 91017 : 71017, true));
  std::memcpy(&result, mj.host(BUF_BASE), sizeof result);
  CHECK(result == (t.regs[4] & 0xff0000ffu));

  t = Trial{};
  t.require_halt = true;
  t.spsr = 0x12345678u; t.regs[1] = BUF_BASE; t.regs[4] = 0xa5b6c7d8u;
  t.regs[13] = STACK;
  t.code = {0xE321F010, 0xE169F004, 0xE14F0000, 0xE5810000,
            0xE7F000F0}; // User mode ignores SPSR writes; undefined trap exits.
  CHECK(run_both(mi, mj, t, a9, a9 ? 91018 : 71018, true));
  std::memcpy(&result, mj.host(BUF_BASE), sizeof result);
  CHECK(result == t.spsr);

  t = Trial{};
  t.require_halt = true;
  t.code = {0xE3A0E055, 0xE92D4000, 0xE3A0E000, 0xE8BD4000,
            0xE584E000}; // push/pop {lr}, then verify the restored value
  for (int i = 0; i < 15; ++i) t.regs[i] = 0;
  t.regs[4] = BUF_BASE; t.regs[13] = STACK;
  CHECK(run_both(mi, mj, t, a9, a9 ? 91007 : 71007, true));
  std::memcpy(&result, mj.host(BUF_BASE), sizeof result);
  CHECK(result == 0x55u && mj.cpu.hot.regs[13] == STACK);

  if (a9) {
    t = Trial{};
    t.require_halt = true;
    t.code = {0xE894000F, 0xE5850000, 0xE5851004, 0xE5852008, 0xE585300C};
    for (int i = 0; i < 15; ++i) t.regs[i] = 0;
    t.regs[4] = BUF_BASE + 0x100; t.regs[5] = BUF_BASE; t.regs[13] = STACK;
    CHECK(run_both(mi, mj, t, true, 91008, true)); // LDMIA {r0-r3}

    t = Trial{};
    t.require_halt = true;
    t.code = {0xE924000F}; // STMDB r4!, {r0-r3}
    for (int i = 0; i < 15; ++i) t.regs[i] = 0x12340000u + i;
    t.regs[4] = BUF_BASE + 0x240; t.regs[13] = STACK;
    CHECK(run_both(mi, mj, t, true, 91009, true));
    CHECK(mj.cpu.hot.regs[4] == BUF_BASE + 0x230);

    t = Trial{};
    t.require_halt = true;
    t.cpsr = 0x80000000u; // LT: exercise the hottest profiled STMIA instruction.
    t.code = {0xB8A101FD}; // STMIALT r1!, {r0,r2-r8}
    t.code_base = 0x020367B4;
    for (int i = 0; i < 15; ++i) t.regs[i] = 0xA5A50000u + i;
    t.regs[1] = BUF_BASE + 0x400; t.regs[13] = STACK;
    CHECK(run_both(mi, mj, t, true, 91012, true));
    CHECK(mj.cpu.hot.regs[1] == BUF_BASE + 0x420);

    t = Trial{};
    t.require_halt = true;
    t.code = {0x0894000F}; // LDMIAEQ; Z is clear, so the transfer is skipped.
    for (int i = 0; i < 15; ++i) t.regs[i] = 0;
    t.regs[4] = BUF_BASE + 0x100; t.regs[13] = STACK;
    CHECK(run_both(mi, mj, t, true, 91010, true));

    t = Trial{};
    t.require_halt = true;
    t.code = {0xE8940003}; // Crossing the 2 KiB mapping page must take fallback.
    for (int i = 0; i < 15; ++i) t.regs[i] = 0;
    t.regs[4] = BUF_BASE + 0x7fc; t.regs[13] = STACK;
    CHECK(run_both(mi, mj, t, true, 91011, true));

    for (u32 mode = 0; mode < 16; ++mode) {
      t = Trial{};
      t.require_halt = true;
      const u32 list = (mode & 1) ? 0x4105u : 0x7a25u;
      t.code = {0xE8000000u | ((mode & 1) << 24) | ((mode & 2) << 22) |
                ((mode & 4) << 19) | ((mode & 8) << 17) | 0x00040000u | list};
      for (int i = 0; i < 15; ++i) t.regs[i] = 0x76540000u + 0x101u * i;
      t.regs[4] = BUF_BASE + 0x600; t.regs[13] = STACK;
      if (mode & 1) t.cpsr = (mode & 2) ? 0x40000000u : 0x80000000u;
      CHECK(run_both(mi, mj, t, true, 91100 + mode, true));
    }
  }

  t = Trial{};
  t.require_halt = true;
  t.code = {0xEB000002, 0xE3A00001, 0xE5890004, 0xE3A00002,
            0xE589E000}; // BL skips a visible store; save LR
  for (int i = 0; i < 15; ++i) t.regs[i] = 0;
  t.regs[9] = BUF_BASE; t.regs[13] = STACK;
  CHECK(run_both(mi, mj, t, a9, a9 ? 91006 : 71006, true));
  std::memcpy(&result, mj.host(BUF_BASE), sizeof result);
  CHECK(result == CODE_BASE + 4);
  std::memcpy(&result, mj.host(BUF_BASE + 4), sizeof result);
  CHECK(result == 0xA4A4A4A4u);

  if (a9) {
    auto passes = [](u32 cond, u32 f) {
      const bool n = f & 8, z = f & 4, c = f & 2, v = f & 1;
      switch (cond) {
      case 0: return z; case 1: return !z; case 2: return c; case 3: return !c;
      case 4: return n; case 5: return !n; case 6: return v; case 7: return !v;
      case 8: return c && !z; case 9: return !c || z; case 10: return n == v;
      case 11: return n != v; case 12: return !z && n == v; default: return z || n != v;
      }
    };
    for (u32 cond = 0; cond < 14; ++cond) {
      for (u32 flags = 0; flags < 16; ++flags) {
        t = Trial{};
        t.require_halt = true;
        t.cpsr = flags << 28;
        t.code = {(cond << 28) | 0x0a000002, 0xE3A00001, 0xEA000001,
                  0xE1A00000, 0xE3A00002, 0xE5890000};
        for (int i = 0; i < 15; ++i) t.regs[i] = 0;
        t.regs[9] = BUF_BASE; t.regs[13] = STACK;
        CHECK(run_both(mi, mj, t, a9, 95000 + cond * 16 + flags, true));
        std::memcpy(&result, mj.host(BUF_BASE), sizeof result);
        CHECK(result == (passes(cond, flags) ? 2u : 1u));
      }
    }
  }
  if (!a9) {
    for (u32 mode = 0; mode < 16; ++mode) {
      t = Trial{};
      t.require_halt = true;
      const u32 list = (mode & 1) ? 0x4105u : 0x7a25u;
      t.code = {0xE8000000u | ((mode & 1) << 24) | ((mode & 2) << 22) |
                ((mode & 4) << 19) | ((mode & 8) << 17) | 0x00040000u | list};
      for (int i = 0; i < 15; ++i) t.regs[i] = 0x76540000u + 0x101u * i;
      t.regs[4] = BUF_BASE + 0x600; t.regs[13] = STACK;
      if (mode & 1) t.cpsr = (mode & 2) ? 0x40000000u : 0x80000000u;
      CHECK(run_both(mi, mj, t, false, 71100 + mode, true));
    }
  }
  jit::detach(mj.nds);
  unsetenv("DS_MIPS_NATIVE_BRANCH");
  unsetenv("DS_MIPS_NATIVE_MEM");
}

__attribute__((noinline)) void native_thumb_smoke(Cpu which) {
  static Machine mi9(Cpu::ARM9), mj9(Cpu::ARM9);
  static Machine mi7(Cpu::ARM7), mj7(Cpu::ARM7);
  Machine& mi = which == Cpu::ARM9 ? mi9 : mi7;
  Machine& mj = which == Cpu::ARM9 ? mj9 : mj7;
  const bool a9 = which == Cpu::ARM9;
  setenv("DS_MIPS_NATIVE", "1", 1);
  setenv("DS_MIPS_NATIVE_ARM7", "1", 1);
  setenv("DS_MIPS_NATIVE_OP", "all", 1);
  setenv("DS_MIPS_NATIVE_BRANCH", "1", 1);
  setenv("DS_MIPS_NATIVE_MEM", "1", 1);
  CHECK(jit::attach(mj.nds, a9, !a9));
  Trial t{};
  t.thumb = true;
  t.require_halt = true;
  t.code = {0x2001, 0x3002, 0x07c1, 0x080a, 0x100b,
            0x6030, 0x6071, 0x60b2, 0x60f3,
            0x2080, 0x2181, 0x2280, 0x400a, 0x6132,
            0x404a, 0x6172, 0x2280, 0x420a, 0x430a,
            0x61b2, 0x4690, 0x4640, 0x61f0, 0x46c0};
  // Immediate moves/add, immediate shifts, AND/EOR/TST/ORR and stores.
  for (int i = 0; i < 15; ++i) t.regs[i] = 0;
  t.regs[6] = BUF_BASE;
  t.regs[13] = STACK;
  CHECK(run_both(mi, mj, t, a9, a9 ? 94001 : 74001, true));
  u32 results[8]{};
  std::memcpy(results, mj.host(BUF_BASE), sizeof results);
  CHECK(results[0] == 3 && results[1] == 0x80000000u && results[2] == 0 &&
        results[3] == 0xffffffffu && results[4] == 0x80 && results[5] == 1 &&
        results[6] == 0x81 && results[7] == 0x81);
  Trial push_pop{};
  push_pop.thumb = true;
  push_pop.require_halt = true;
  push_pop.code = {0xB401, 0x2000, 0xBC02, 0x6031}; // PUSH {r0}; POP {r1}; STR r1,[r6]
  push_pop.regs[0] = 0x12345678u;
  push_pop.regs[6] = BUF_BASE;
  push_pop.regs[13] = STACK;
  CHECK(run_both(mi, mj, push_pop, a9, a9 ? 94002 : 74002, true));
  std::memcpy(&results[0], mj.host(BUF_BASE), sizeof(u32));
  CHECK(results[0] == 0x12345678u);
  Trial pop_pc{};
  pop_pc.thumb = true;
  pop_pc.require_halt = true;
  pop_pc.code = {0x9000, 0xBD00}; // STR r0,[sp]; POP {pc} to the halt trampoline.
  pop_pc.regs[0] = (CODE_BASE + 4) | (a9 ? 1u : 0u);
  pop_pc.regs[13] = BUF_BASE;
  CHECK(run_both(mi, mj, pop_pc, a9, a9 ? 94006 : 74006, true));
  if (!a9) {
    Trial thumb_pop_pc{};
    thumb_pop_pc.thumb = true;
    thumb_pop_pc.require_halt = true;
    thumb_pop_pc.code = {0xA101, 0x3101, 0x9101, 0xBD01, 0x466A, 0x601A};
    thumb_pop_pc.regs[3] = BUF_BASE + 0x100;
    thumb_pop_pc.regs[13] = BUF_BASE;
    CHECK(run_both(mi, mj, thumb_pop_pc, false, 74007, true));
    std::memcpy(&results[0], mj.host(BUF_BASE + 0x100), sizeof(u32));
    CHECK(results[0] == BUF_BASE + 8);
    Trial iwram_push_pop{};
    iwram_push_pop.thumb = true;
    iwram_push_pop.require_halt = true;
    iwram_push_pop.code = {0xB40F, 0x2000, 0x2100, 0x2200, 0x2300, 0xBC0F,
                           0x6030, 0x6071, 0x60B2, 0x60F3};
    iwram_push_pop.regs[0] = 0x12345678u;
    iwram_push_pop.regs[1] = 0x87654321u;
    iwram_push_pop.regs[2] = 0xA5A55A5Au;
    iwram_push_pop.regs[3] = 0x5A5AA5A5u;
    iwram_push_pop.regs[6] = BUF_BASE;
    iwram_push_pop.regs[13] = 0x0380FF00u;
    CHECK(run_both(mi, mj, iwram_push_pop, false, 74003, true));
    std::memcpy(results, mj.host(BUF_BASE), sizeof(u32) * 4);
    CHECK(results[0] == 0x12345678u && results[1] == 0x87654321u &&
          results[2] == 0xA5A55A5Au && results[3] == 0x5A5AA5A5u);
    Trial arm7_thumb_ldrsb_half{};
    arm7_thumb_ldrsb_half.thumb = true;
    arm7_thumb_ldrsb_half.require_halt = true;
    arm7_thumb_ldrsb_half.code = {0x2380, 0x704B, 0x2001, 0x5E0A, 0x6032};
    arm7_thumb_ldrsb_half.regs[1] = BUF_BASE;
    arm7_thumb_ldrsb_half.regs[6] = BUF_BASE + 4;
    arm7_thumb_ldrsb_half.regs[13] = STACK;
    CHECK(run_both(mi, mj, arm7_thumb_ldrsb_half, false, 74005, true));
    std::memcpy(&results[0], mj.host(BUF_BASE + 4), sizeof(u32));
    CHECK(results[0] == 0xffffff80u);
    Trial thumb_stm_ldm{};
    thumb_stm_ldm.thumb = true;
    thumb_stm_ldm.require_halt = true;
    thumb_stm_ldm.code = {0xC40F, 0x2000, 0x2100, 0x2200, 0x2300,
                          0x3C10, 0xCC0F, 0x6030, 0x6071, 0x60B2, 0x60F3};
    thumb_stm_ldm.regs[0] = 0x12345678u;
    thumb_stm_ldm.regs[1] = 0x87654321u;
    thumb_stm_ldm.regs[2] = 0xA5A55A5Au;
    thumb_stm_ldm.regs[3] = 0x5A5AA5A5u;
    thumb_stm_ldm.regs[4] = BUF_BASE + 0x400;
    thumb_stm_ldm.regs[6] = BUF_BASE;
    thumb_stm_ldm.regs[13] = STACK;
    CHECK(run_both(mi, mj, thumb_stm_ldm, false, 74004, true));
    std::memcpy(results, mj.host(BUF_BASE), sizeof(u32) * 4);
    CHECK(results[0] == 0x12345678u && results[1] == 0x87654321u &&
          results[2] == 0xA5A55A5Au && results[3] == 0x5A5AA5A5u);
  }
  for (u32 op = 0; op < 16; ++op) {
    const u32 max_amount = (op >= 2 && op <= 4) || op == 7 ? 5 : 1;
    for (u32 i = 0; i < max_amount; ++i) {
      static constexpr u32 amounts[] = {0, 1, 32, 33, 255};
      Trial alu{};
      alu.thumb = true;
      alu.require_halt = true;
      alu.code = {static_cast<u16>(0x4000u | (op << 6) | (1u << 3) | 2u)};
      alu.regs[1] = (op >= 2 && op <= 4) || op == 7 ? amounts[i] : 0xf1234567u;
      alu.regs[2] = 0x81234567u;
      alu.regs[13] = STACK;
      CHECK(run_both(mi, mj, alu, a9,
                     (a9 ? 95000 : 75000) + op * 8 + i, true));
    }
  }
  if (a9) {
    Trial thumb_stm_ldm_a9{};
    thumb_stm_ldm_a9.thumb = true;
    thumb_stm_ldm_a9.require_halt = true;
    thumb_stm_ldm_a9.code = {0xC40F, 0x2000, 0x2100, 0x2200, 0x2300,
                             0x3C10, 0xCC0F, 0x6030, 0x6071, 0x60B2, 0x60F3};
    thumb_stm_ldm_a9.regs[0] = 0x12345678u;
    thumb_stm_ldm_a9.regs[1] = 0x87654321u;
    thumb_stm_ldm_a9.regs[2] = 0xA5A55A5Au;
    thumb_stm_ldm_a9.regs[3] = 0x5A5AA5A5u;
    thumb_stm_ldm_a9.regs[4] = BUF_BASE + 0x400;
    thumb_stm_ldm_a9.regs[6] = BUF_BASE;
    thumb_stm_ldm_a9.regs[13] = STACK;
    CHECK(run_both(mi, mj, thumb_stm_ldm_a9, true, 94008, true));
    u32 block_results[4]{};
    std::memcpy(block_results, mj.host(BUF_BASE), sizeof block_results);
    CHECK(block_results[0] == 0x12345678u && block_results[1] == 0x87654321u &&
          block_results[2] == 0xA5A55A5Au && block_results[3] == 0x5A5AA5A5u);
    Trial mem{};
    mem.thumb = true;
    mem.require_halt = true;
    mem.code = {0x2080, 0x7008, 0x56CA, 0x604A, 0x2001, 0x03C0,
                0x8048, 0x2302, 0x5ECA, 0x608A, 0x5ACA, 0x60CA,
                0x230D, 0x58CA, 0x614A, 0x2001, 0x9001, 0x9A01, 0x618A};
    for (int r = 0; r < 15; ++r) mem.regs[r] = 0;
    mem.regs[1] = BUF_BASE;
    mem.regs[13] = STACK;
    CHECK(run_both(mi, mj, mem, true, 94004, true));
    u32 memory_results[7]{};
    std::memcpy(memory_results, mj.host(BUF_BASE), sizeof memory_results);
    CHECK(memory_results[1] == 0xffffff80u); // LDRSB
    CHECK(memory_results[2] == 0xffff8000u); // LDRSH
    CHECK(memory_results[3] == 0x00008000u); // LDRH
    CHECK(memory_results[5] == 0x00000080u); // unaligned LDR rotation
    CHECK(memory_results[6] == 0x00000001u); // SP-relative store/load roundtrip
    Trial tagged{};
    tagged.thumb = true;
    tagged.require_halt = true;
    tagged.code = {0x205A, 0x7008, 0x5CCA, 0x6032};
    for (int r = 0; r < 15; ++r) tagged.regs[r] = 0;
    tagged.regs[1] = CODE_BASE + 0x800;
    tagged.regs[6] = BUF_BASE;
    tagged.regs[13] = STACK;
    CHECK(run_both(mi, mj, tagged, true, 94005, true));
    std::memcpy(&results[0], mj.host(BUF_BASE), sizeof(u32));
    CHECK(results[0] == 0x5Au); // tagged/code memory goes through SMC helpers
    auto enable_dtcm = [](Machine& m) {
      m.cpu.cp15_control |= 1u << 16;
      m.cpu.cp15_dtcm = 0x027C0000u | (5u << 1); // 16 KB DTCM at 0x027C0000
      m.nds.bus.update_tcm(m.cpu);
    };
    enable_dtcm(mi);
    enable_dtcm(mj);
    Trial dtcm_push_pop{};
    dtcm_push_pop.thumb = true;
    dtcm_push_pop.require_halt = true;
    dtcm_push_pop.code = {0xB40F, 0x2000, 0x2100, 0x2200, 0x2300, 0xBC0F,
                          0x6030, 0x6071, 0x60B2, 0x60F3};
    dtcm_push_pop.regs[0] = 0x12345678u;
    dtcm_push_pop.regs[1] = 0x87654321u;
    dtcm_push_pop.regs[2] = 0xA5A55A5Au;
    dtcm_push_pop.regs[3] = 0x5A5AA5A5u;
    dtcm_push_pop.regs[6] = BUF_BASE;
    dtcm_push_pop.regs[13] = 0x027C3F00u;
    CHECK(run_both(mi, mj, dtcm_push_pop, true, 94007, true));
    std::memcpy(results, mj.host(BUF_BASE), sizeof(u32) * 4);
    CHECK(results[0] == 0x12345678u && results[1] == 0x87654321u &&
          results[2] == 0xA5A55A5Au && results[3] == 0x5A5AA5A5u);
    Trial arm_mem{};
    arm_mem.require_halt = true;
    arm_mem.code = {0xE3A00080, 0xE5C10000, 0xE5D12000, 0xE5812004,
                    0xE5913004, 0xE5813008, 0xE3A05001, 0xE7916105,
                    0xE5816010, 0xE3500000, 0x05914000, 0xE581400C};
    for (int r = 0; r < 15; ++r) arm_mem.regs[r] = 0;
    arm_mem.regs[1] = BUF_BASE;
    arm_mem.regs[4] = 0x55;
    arm_mem.regs[13] = STACK;
    CHECK(run_both(mi, mj, arm_mem, true, 94006, true));
    u32 arm_memory_results[5]{};
    std::memcpy(arm_memory_results, mj.host(BUF_BASE), sizeof arm_memory_results);
    CHECK(arm_memory_results[1] == 0x80u);
    CHECK(arm_memory_results[2] == 0x80u);
    CHECK(arm_memory_results[3] == 0x55u); // failed EQ LDR did not write r4
    CHECK(arm_memory_results[4] == 0x80u); // shifted register-offset LDR/STR
    Trial arm_half{};
    arm_half.require_halt = true;
    arm_half.code = {0xE3A00001, 0xE1A00780, 0xE1C100B2, 0xE1D120B2,
                     0xE5812004, 0xE1D130D3, 0xE5813008, 0xE1D140F2,
                     0xE581400C};
    for (int r = 0; r < 15; ++r) arm_half.regs[r] = 0;
    arm_half.regs[1] = BUF_BASE;
    arm_half.regs[13] = STACK;
    CHECK(run_both(mi, mj, arm_half, true, 94007, true));
    std::memcpy(arm_memory_results, mj.host(BUF_BASE), sizeof arm_memory_results);
    CHECK(arm_memory_results[1] == 0x00008000u);
    CHECK(arm_memory_results[2] == 0xffffff80u);
    CHECK(arm_memory_results[3] == 0xffff8000u);
    for (u32 amount : {0u, 1u, 32u, 255u}) {
      Trial reg_shift{};
      reg_shift.require_halt = true;
      reg_shift.code = {0xE1A02013, 0xE1A04035, 0xE1A06057, 0xE1A08079,
                        0xE5812000, 0xE5814004, 0xE5816008, 0xE581800C};
      for (int r = 0; r < 15; ++r) reg_shift.regs[r] = 0;
      reg_shift.regs[0] = amount;
      reg_shift.regs[1] = BUF_BASE;
      reg_shift.regs[3] = reg_shift.regs[5] = reg_shift.regs[7] =
          reg_shift.regs[9] = 0x80000001u;
      reg_shift.regs[13] = STACK;
      CHECK(run_both(mi, mj, reg_shift, true, 94009 + amount, true));
    }
    Trial post{};
    post.require_halt = true;
    post.code = {0xE3A00066, 0xE5810000, 0xE4912004,
                 0xE5812000, 0xE5811004};
    for (int r = 0; r < 15; ++r) post.regs[r] = 0;
    post.regs[1] = BUF_BASE;
    post.regs[13] = STACK;
    CHECK(run_both(mi, mj, post, true, 94008, true));
    std::memcpy(arm_memory_results, mj.host(BUF_BASE), sizeof arm_memory_results);
    CHECK(arm_memory_results[0] == 0x66u);
    CHECK(arm_memory_results[1] == 0x66u);
    CHECK(arm_memory_results[2] == BUF_BASE + 4); // post-index base writeback
    Trial cp15_nop{};
    cp15_nop.require_halt = true;
    cp15_nop.code = {0xEE07CF9A, 0xEE070F3E, 0x0E07CF9A};
    for (int r = 0; r < 15; ++r) cp15_nop.regs[r] = 0;
    cp15_nop.regs[13] = STACK;
    CHECK(run_both(mi, mj, cp15_nop, true, 94010, true));
    Trial bx{};
    bx.require_halt = true;
    bx.code = {0xE3A00001, 0xE12FFF13, 0xE3A01077, 0xE3A02066,
               0xE5842000};
    for (int r = 0; r < 15; ++r) bx.regs[r] = 0;
    bx.regs[3] = CODE_BASE + 12;
    bx.regs[4] = BUF_BASE;
    bx.regs[13] = STACK;
    CHECK(run_both(mi, mj, bx, true, 94011, true));
    std::memcpy(&results[0], mj.host(BUF_BASE), sizeof(u32));
    CHECK(results[0] == 0x66u && mj.cpu.hot.regs[1] == 0);
    bx.code[1] = 0x012FFF13; // EQ BX, but Z is clear: execute fall-through.
    CHECK(run_both(mi, mj, bx, true, 94012, true));
    CHECK(mj.cpu.hot.regs[1] == 0x77u);
    Trial multi{};
    multi.require_halt = true;
    multi.code = {0xE8A10001, 0xE5841000}; // STMIA with writeback, then save Rn
    for (int r = 0; r < 15; ++r) multi.regs[r] = 0;
    multi.regs[0] = 0x66u;
    multi.regs[1] = BUF_BASE;
    multi.regs[4] = BUF_BASE + 8;
    multi.regs[13] = STACK;
    CHECK(run_both(mi, mj, multi, true, 94013, true));
    std::memcpy(&results[0], mj.host(BUF_BASE), sizeof(u32));
    CHECK(results[0] == 0x66u);
    std::memcpy(&results[0], mj.host(BUF_BASE + 8), sizeof(u32));
    CHECK(results[0] == BUF_BASE + 4);
    multi.code = {0xE8D10001, 0xE5840004}; // LDMIA^ then store the loaded value
    multi.regs[1] = BUF_BASE;
    multi.regs[4] = BUF_BASE;
    CHECK(run_both(mi, mj, multi, true, 94014, true));
    std::memcpy(&results[0], mj.host(BUF_BASE + 4), sizeof(u32));
    CHECK(results[0] == 0xA5A5A5A5u);
    Trial mul{};
    mul.require_halt = true;
    mul.code = {0xE0020493, 0xE5812000}; // MUL r2,r3,r4; STR r2,[r1]
    for (int r = 0; r < 15; ++r) mul.regs[r] = 0;
    mul.regs[1] = BUF_BASE;
    mul.regs[3] = 0x12345678u;
    mul.regs[4] = 0x10u;
    mul.regs[13] = STACK;
    CHECK(run_both(mi, mj, mul, true, 94015, true));
    std::memcpy(&results[0], mj.host(BUF_BASE), sizeof(u32));
    CHECK(results[0] == 0x23456780u);
    Trial umull{};
    umull.require_halt = true;
    umull.code = {0xE0812394, 0xE5852000, 0xE5851004}; // UMULL r2,r1,r4,r3
    for (int r = 0; r < 15; ++r) umull.regs[r] = 0;
    umull.regs[3] = 0x12345678u;
    umull.regs[4] = 0x10u;
    umull.regs[5] = BUF_BASE;
    umull.regs[13] = STACK;
    CHECK(run_both(mi, mj, umull, true, 94018, true));
    std::memcpy(results, mj.host(BUF_BASE), sizeof(u32) * 2);
    CHECK(results[0] == 0x23456780u && results[1] == 1u);
    const struct { u32 lhs, rhs, lo, hi; } signed_products[] = {
        {0xffffffffu, 2u, 0xfffffffeu, 0xffffffffu},
        {0x80000000u, 0x80000000u, 0u, 0x40000000u},
        {0x7fffffffu, 2u, 0xfffffffeu, 0u},
        {0x80000000u, 2u, 0u, 0xffffffffu},
    };
    for (u32 i = 0; i < sizeof signed_products / sizeof signed_products[0]; ++i) {
      for (u32 form = 0; form < 2; ++form) {
        Trial t{};
        t.require_halt = true;
        t.code = {form ? 0xe0c01392u : 0xe0c10092u,
                  form ? 0xe5851000u : 0xe5850000u,
                  form ? 0xe5850004u : 0xe5851004u};
        t.regs[0] = form ? 0x11111111u : signed_products[i].lhs;
        t.regs[1] = form ? 0x22222222u : 0;
        t.regs[2] = signed_products[i].rhs;
        t.regs[3] = form ? signed_products[i].lhs : 0;
        t.regs[5] = BUF_BASE;
        t.regs[13] = STACK;
        CHECK(run_both(mi, mj, t, true, 94030 + i * 2 + form, true));
        std::memcpy(results, mj.host(BUF_BASE), sizeof(u32) * 2);
        CHECK(results[0] == signed_products[i].lo && results[1] == signed_products[i].hi);
      }
    }
    static constexpr u32 long_mul[] = {
        0xE0912394u, 0xE0B12394u, 0xE0D12394u, 0xE0F12394u
    }; // UMULLS, UMLALS, SMULLS, SMLALS r2,r1,r4,r3
    static constexpr u32 long_mul_hi[] = {1u, 0x80000001u, 0xffffffffu, 0x7fffffffu};
    static constexpr u32 long_mul_lo[] = {0xfffffffeu, 0xfffffffdu,
                                          0xfffffffeu, 0xfffffffdu};
    static constexpr u32 long_mul_flags[] = {0x30000000u, 0xb0000000u,
                                             0xb0000000u, 0x30000000u};
    for (u32 i = 0; i < 4; ++i) {
      Trial t{};
      t.require_halt = true;
      t.cpsr = 0xf0000000u;
      t.code = {long_mul[i], 0xE5852000u, 0xE5851004u,
                0xE10F6000u, 0xE5856008u};
      t.regs[1] = 0x7fffffffu;
      t.regs[2] = 0xffffffffu;
      t.regs[3] = 0xffffffffu;
      t.regs[4] = 2u;
      t.regs[5] = BUF_BASE;
      t.regs[13] = STACK;
      CHECK(run_both(mi, mj, t, true, 94020 + i, true));
      std::memcpy(results, mj.host(BUF_BASE), sizeof(u32) * 3);
      CHECK(results[0] == long_mul_lo[i] && results[1] == long_mul_hi[i]);
      CHECK((results[2] & 0xf0000000u) == long_mul_flags[i]);
    }
    const struct { u32 instr, a, b, acc, out, cpsr, flags; } mla_cases[] = {
        {0xe0225493u, 0x12345678u, 0x10u, 0xf0000001u, 0x13456781u,
         0x30000000u, 0x30000000u},
        {0xe0325493u, 0u, 0u, 0u, 0u, 0x10000000u, 0x50000000u},
    };
    for (u32 i = 0; i < sizeof mla_cases / sizeof mla_cases[0]; ++i) {
      Trial t{};
      t.require_halt = true;
      t.cpsr = mla_cases[i].cpsr;
      t.code = {mla_cases[i].instr, 0xe5862000u, 0xe10f7000u, 0xe5867004u};
      t.regs[2] = 0xdeadbeefu;
      t.regs[3] = mla_cases[i].a;
      t.regs[4] = mla_cases[i].b;
      t.regs[5] = mla_cases[i].acc;
      t.regs[6] = BUF_BASE;
      t.regs[13] = STACK;
      CHECK(run_both(mi, mj, t, true, 94040 + i, true));
      std::memcpy(results, mj.host(BUF_BASE), sizeof(u32) * 2);
      CHECK(results[0] == mla_cases[i].out);
      CHECK((results[1] & 0xf0000000u) == mla_cases[i].flags);
    }
    Trial mvn_zero{};
    mvn_zero.require_halt = true;
    mvn_zero.code = {0xE3E00000, 0xE5810000}; // MVN r0,#0; STR r0,[r1]
    for (int r = 0; r < 15; ++r) mvn_zero.regs[r] = 0;
    mvn_zero.regs[1] = BUF_BASE;
    mvn_zero.regs[13] = STACK;
    CHECK(run_both(mi, mj, mvn_zero, true, 94016, true));
    std::memcpy(&results[0], mj.host(BUF_BASE), sizeof(u32));
    CHECK(results[0] == ~u32{0});
    Trial mrs{};
    mrs.require_halt = true;
    mrs.code = {0xE10F0000, 0xE5810000}; // MRS r0,CPSR; STR r0,[r1]
    for (int r = 0; r < 15; ++r) mrs.regs[r] = 0;
    mrs.regs[1] = BUF_BASE;
    mrs.regs[13] = STACK;
    CHECK(run_both(mi, mj, mrs, true, 94017, true));
    std::memcpy(&results[0], mj.host(BUF_BASE), sizeof(u32));
    CHECK(results[0] == mj.cpu.hot.cpsr);
    Trial complements{};
    complements.require_halt = true;
    complements.code = {0xE1E00003, 0xE5840000, 0xE1C10002, 0xE5840004,
                        0xE1F00003, 0xE5840008};
    for (int r = 0; r < 15; ++r) complements.regs[r] = 0;
    complements.regs[1] = 0xFFFFFFF0u;
    complements.regs[2] = 0x0Fu;
    complements.regs[3] = 0;
    complements.regs[4] = BUF_BASE;
    complements.regs[13] = STACK;
    CHECK(run_both(mi, mj, complements, true, 94019, true));
    std::memcpy(results, mj.host(BUF_BASE), sizeof(u32) * 3);
    CHECK(results[0] == ~u32{0} && results[1] == 0xFFFFFFF0u &&
          results[2] == ~u32{0});
  } else {
    Trial arm7_mem{};
    arm7_mem.require_halt = true;
    arm7_mem.code = {0xE3A0005A, 0xE5810000, 0xE5912000, 0xE5812004};
    for (int r = 0; r < 15; ++r) arm7_mem.regs[r] = 0;
    arm7_mem.regs[1] = BUF_BASE;
    arm7_mem.regs[13] = STACK;
    CHECK(run_both(mi, mj, arm7_mem, false, 74004, true));
    u32 arm7_words[2]{};
    std::memcpy(arm7_words, mj.host(BUF_BASE), sizeof arm7_words);
    CHECK(arm7_words[0] == 0x5Au && arm7_words[1] == 0x5Au);
    Trial arm7_thumb_mem{};
    arm7_thumb_mem.thumb = true;
    arm7_thumb_mem.require_halt = true;
    arm7_thumb_mem.code = {0x205A, 0x6008, 0x680A, 0x604A};
    for (int r = 0; r < 15; ++r) arm7_thumb_mem.regs[r] = 0;
    arm7_thumb_mem.regs[1] = BUF_BASE;
    arm7_thumb_mem.regs[13] = STACK;
    CHECK(run_both(mi, mj, arm7_thumb_mem, false, 74005, true));
    std::memcpy(arm7_words, mj.host(BUF_BASE), sizeof arm7_words);
    CHECK(arm7_words[0] == 0x5Au && arm7_words[1] == 0x5Au);
    Trial bx{};
    bx.require_halt = true;
    bx.code = {0xE3A00001, 0xE12FFF13, 0xE3A01077, 0xE3A02066,
               0xE5842000};
    for (int r = 0; r < 15; ++r) bx.regs[r] = 0;
    bx.regs[3] = CODE_BASE + 12;
    bx.regs[4] = BUF_BASE;
    bx.regs[13] = STACK;
    CHECK(run_both(mi, mj, bx, false, 74006, true));
    std::memcpy(arm7_words, mj.host(BUF_BASE), sizeof(u32));
    CHECK(arm7_words[0] == 0x66u);
    Trial multi{};
    multi.require_halt = true;
    multi.code = {0xE8A10001, 0xE5841000};
    for (int r = 0; r < 15; ++r) multi.regs[r] = 0;
    multi.regs[0] = 0x66u;
    multi.regs[1] = BUF_BASE;
    multi.regs[4] = BUF_BASE + 8;
    multi.regs[13] = STACK;
    CHECK(run_both(mi, mj, multi, false, 74007, true));
    std::memcpy(arm7_words, mj.host(BUF_BASE), sizeof(u32));
    CHECK(arm7_words[0] == 0x66u);
    std::memcpy(arm7_words, mj.host(BUF_BASE + 8), sizeof(u32));
    CHECK(arm7_words[0] == BUF_BASE + 4);
    multi.code = {0xE8D10001, 0xE5840004};
    multi.regs[1] = BUF_BASE;
    multi.regs[4] = BUF_BASE;
    CHECK(run_both(mi, mj, multi, false, 74008, true));
    std::memcpy(arm7_words, mj.host(BUF_BASE + 4), sizeof(u32));
    CHECK(arm7_words[0] == 0xA5A5A5A5u);
    Trial mrs{};
    mrs.require_halt = true;
    mrs.code = {0xE10F0000, 0xE5810000};
    for (int r = 0; r < 15; ++r) mrs.regs[r] = 0;
    mrs.regs[1] = BUF_BASE;
    mrs.regs[13] = STACK;
    CHECK(run_both(mi, mj, mrs, false, 74009, true));
    std::memcpy(arm7_words, mj.host(BUF_BASE), sizeof(u32));
    CHECK(arm7_words[0] == mj.cpu.hot.cpsr);
    Trial mvn{};
    mvn.require_halt = true;
    mvn.code = {0xE1E01003, 0xE5841000};
    for (int r = 0; r < 15; ++r) mvn.regs[r] = 0;
    mvn.regs[3] = 0;
    mvn.regs[4] = BUF_BASE;
    mvn.regs[13] = STACK;
    CHECK(run_both(mi, mj, mvn, false, 74010, true));
    std::memcpy(arm7_words, mj.host(BUF_BASE), sizeof(u32));
    CHECK(arm7_words[0] == ~u32{0});
  }
  Trial branch{};
  branch.thumb = true;
  branch.require_halt = true;
  branch.code = {0xE001, 0x2001, 0x6048, 0x2003, 0x6008};
  for (int r = 0; r < 15; ++r) branch.regs[r] = 0;
  branch.regs[1] = BUF_BASE;
  branch.regs[13] = STACK;
  CHECK(run_both(mi, mj, branch, a9, a9 ? 94002 : 74002, true));
  std::memcpy(results, mj.host(BUF_BASE), 8);
  CHECK(results[0] == 3 && results[1] == 0xA4A4A4A4u);
  Trial call{};
  call.thumb = true;
  call.require_halt = true;
  call.code = {0xF000, 0xF805, 0x2001, 0x600A, 0xE003,
               0x46C0, 0x46C0, 0x2203, 0x4770};
  for (int r = 0; r < 15; ++r) call.regs[r] = 0;
  call.regs[1] = BUF_BASE;
  call.regs[13] = STACK;
  CHECK(run_both(mi, mj, call, a9, a9 ? 94003 : 74003, true));
  std::memcpy(results, mj.host(BUF_BASE), sizeof(u32));
  CHECK(results[0] == 3);
  for (u32 cond = 0; cond < 14; ++cond) {
    for (u32 flags = 0; flags < 16; ++flags) {
      Trial conditional{};
      conditional.thumb = true;
      conditional.require_halt = true;
      conditional.cpsr = flags << 28;
      conditional.code = {0xD001u | (cond << 8), 0x2001, 0xE000, 0x2002, 0x6008};
      for (int r = 0; r < 15; ++r) conditional.regs[r] = 0;
      conditional.regs[1] = BUF_BASE;
      conditional.regs[13] = STACK;
      CHECK(run_both(mi, mj, conditional, a9,
                     (a9 ? 94500 : 74500) + cond * 16 + flags, true));
      std::memcpy(results, mj.host(BUF_BASE), sizeof(u32));
      const bool n = flags & 8, z = flags & 4, c = flags & 2, v = flags & 1;
      const bool take = cond == 0 ? z : cond == 1 ? !z : cond == 2 ? c : cond == 3 ? !c :
                        cond == 4 ? n : cond == 5 ? !n : cond == 6 ? v : cond == 7 ? !v :
                        cond == 8 ? c && !z : cond == 9 ? !c || z : cond == 10 ? n == v :
                        cond == 11 ? n != v : cond == 12 ? !z && n == v : z || n != v;
      CHECK(results[0] == (take ? 2u : 1u));
    }
  }
  for (u32 instr : {0x1888u, 0x1a88u}) {
    for (u32 pattern = 0; pattern < 4; ++pattern) {
      Trial arith{};
      arith.thumb = true;
      arith.require_halt = true;
      arith.cpsr = pattern << 28;
      arith.code = {instr, 0x6030};
      for (int r = 0; r < 15; ++r) arith.regs[r] = 0;
      arith.regs[1] = pattern & 1 ? 0x80000000u : 0x7fffffffu;
      arith.regs[2] = pattern & 2 ? 0xffffffffu : 0x80000001u;
      arith.regs[6] = BUF_BASE;
      arith.regs[13] = STACK;
      CHECK(run_both(mi, mj, arith, a9,
                     (a9 ? 94800 : 74800) + instr + pattern, true));
    }
  }
  for (u32 instr : {0x4208u, 0x4288u, 0x42c8u}) {
    for (u32 flags = 0; flags < 16; ++flags) {
      Trial cmp{};
      cmp.thumb = true;
      cmp.require_halt = true;
      cmp.cpsr = flags << 28;
      cmp.code = {instr};
      for (int r = 0; r < 15; ++r) cmp.regs[r] = 0;
      cmp.regs[0] = flags & 1 ? 0x80000000u : 0x7fffffffu;
      cmp.regs[1] = flags & 2 ? 0xffffffffu : 0x80000001u;
      cmp.regs[13] = STACK;
      CHECK(run_both(mi, mj, cmp, a9,
                     (a9 ? 94900 : 74900) + instr + flags, true));
    }
  }
  for (u32 instr : {0x4148u, 0x4188u}) {
    for (u32 flags = 0; flags < 16; ++flags) {
      Trial carry{};
      carry.thumb = true;
      carry.require_halt = true;
      carry.cpsr = flags << 28;
      carry.code = {instr};
      for (int r = 0; r < 15; ++r) carry.regs[r] = 0;
      carry.regs[0] = flags & 1 ? 0xffffffffu : 0x7fffffffu;
      carry.regs[1] = flags & 2 ? 0x80000000u : 1;
      carry.regs[13] = STACK;
      CHECK(run_both(mi, mj, carry, a9,
                     (a9 ? 94970 : 74970) + instr + flags, true));
    }
  }
  Trial high_add{};
  high_add.thumb = true;
  high_add.require_halt = true;
  high_add.code = {0x4480, 0x4640, 0x6030}; // ADD r8,r0; MOV r0,r8; STR r0,[r6]
  for (int r = 0; r < 15; ++r) high_add.regs[r] = 0;
  high_add.regs[0] = 1;
  high_add.regs[6] = BUF_BASE;
  high_add.regs[8] = 0x7fffffffu;
  high_add.regs[13] = STACK;
  CHECK(run_both(mi, mj, high_add, a9, a9 ? 94901 : 74901, true));
  std::memcpy(results, mj.host(BUF_BASE), sizeof(u32));
  CHECK(results[0] == 0x80000000u);
  Trial pc_add{};
  pc_add.thumb = true;
  pc_add.require_halt = true;
  pc_add.code = {0xa001, 0x6008}; // ADD r0,PC,#4; STR r0,[r1]
  for (int r = 0; r < 15; ++r) pc_add.regs[r] = 0;
  pc_add.regs[1] = BUF_BASE;
  pc_add.regs[13] = STACK;
  CHECK(run_both(mi, mj, pc_add, a9, a9 ? 94902 : 74902, true));
  std::memcpy(results, mj.host(BUF_BASE), sizeof(u32));
  CHECK(results[0] == CODE_BASE + 8);
  Trial literal{};
  literal.thumb = true;
  literal.require_halt = true;
  literal.code = {0x4801, 0x6008, 0xE002, 0x46C0, 0x5678, 0x1234};
  for (int r = 0; r < 15; ++r) literal.regs[r] = 0;
  literal.regs[1] = BUF_BASE;
  literal.regs[13] = STACK;
  const u64 fallback_before = jit::stats().instrs_fallback;
  CHECK(run_both(mi, mj, literal, a9, a9 ? 94905 : 74905, true));
  if (!a9) CHECK(jit::stats().instrs_fallback == fallback_before);
  std::memcpy(results, mj.host(BUF_BASE), sizeof(u32));
  CHECK(results[0] == 0x12345678u);
  Trial thumb_pop_pc{};
  thumb_pop_pc.thumb = true;
  thumb_pop_pc.require_halt = true;
  thumb_pop_pc.code = {0x6030, 0xbd00, 0x4608, 0x4700}; // STR r0,[r6]; POP {pc}; MOV r0,r1; BX r0
  for (int r = 0; r < 15; ++r) thumb_pop_pc.regs[r] = 0;
  thumb_pop_pc.regs[0] = CODE_BASE + 5;
  thumb_pop_pc.regs[1] = HALT_STUB;
  thumb_pop_pc.regs[6] = STACK;
  thumb_pop_pc.regs[13] = STACK;
  const u64 pop_fallback_before = jit::stats().instrs_fallback;
  CHECK(run_both(mi, mj, thumb_pop_pc, a9, a9 ? 94906 : 74906, true));
  if (!a9) CHECK(jit::stats().instrs_fallback == pop_fallback_before);
  Trial sp_add{};
  sp_add.thumb = true;
  sp_add.require_halt = true;
  sp_add.code = {0xa801, 0x6008}; // ADD r0,SP,#4; STR r0,[r1]
  for (int r = 0; r < 15; ++r) sp_add.regs[r] = 0;
  sp_add.regs[1] = BUF_BASE;
  sp_add.regs[13] = STACK;
  CHECK(run_both(mi, mj, sp_add, a9, a9 ? 94903 : 74903, true));
  std::memcpy(results, mj.host(BUF_BASE), sizeof(u32));
  CHECK(results[0] == STACK + 4);
  for (u32 ins : {0xb001u, 0xb081u}) {
    Trial adjust{};
    adjust.thumb = true;
    adjust.require_halt = true;
    adjust.code = {ins};
    for (int r = 0; r < 15; ++r) adjust.regs[r] = 0;
    adjust.regs[13] = STACK;
    CHECK(run_both(mi, mj, adjust, a9, (a9 ? 94904 : 74904) + ins, true));
    CHECK(mj.cpu.hot.regs[13] == (ins & 0x80 ? STACK - 4 : STACK + 4));
  }
  for (u32 flags = 0; flags < 16; ++flags) {
    Trial high_cmp{};
    high_cmp.thumb = true;
    high_cmp.require_halt = true;
    high_cmp.cpsr = flags << 28;
    high_cmp.code = {0x4588}; // CMP r8,r1
    for (int r = 0; r < 15; ++r) high_cmp.regs[r] = 0;
    high_cmp.regs[8] = flags & 1 ? 0x80000000u : 0x7fffffffu;
    high_cmp.regs[1] = flags & 2 ? 0xffffffffu : 0x80000001u;
    high_cmp.regs[13] = STACK;
    CHECK(run_both(mi, mj, high_cmp, a9,
                   (a9 ? 94920 : 74920) + flags, true));
  }
  for (u32 value : {0u, 1u, 0x7fffffffu, 0x80000000u, 0xffffffffu}) {
    Trial cmp_imm{};
    cmp_imm.thumb = true;
    cmp_imm.require_halt = true;
    cmp_imm.code = {0x2801}; // CMP r0,#1
    for (int r = 0; r < 15; ++r) cmp_imm.regs[r] = 0;
    cmp_imm.regs[0] = value;
    cmp_imm.regs[13] = STACK;
    CHECK(run_both(mi, mj, cmp_imm, a9,
                   (a9 ? 94950 : 74950) + value % 5, true));
  }
  const struct { u32 instr, input, output, dst; } arithmetic[] = {
      {0x3001, 0xffffffffu, 0, 0}, {0x3001, 0x7fffffffu, 0x80000000u, 0},
      {0x3801, 0, 0xffffffffu, 0}, {0x3801, 0x80000000u, 0x7fffffffu, 0},
      {0x1c41, 0x7fffffffu, 0x80000000u, 1}, {0x1e41, 0, 0xffffffffu, 1}};
  for (u32 i = 0; i < sizeof arithmetic / sizeof arithmetic[0]; ++i) {
    Trial arith{};
    arith.thumb = true;
    arith.require_halt = true;
    arith.code = {arithmetic[i].instr, 0x6030u + arithmetic[i].dst};
    for (int r = 0; r < 15; ++r) arith.regs[r] = 0;
    arith.regs[0] = arithmetic[i].input;
    arith.regs[6] = BUF_BASE;
    arith.regs[13] = STACK;
    CHECK(run_both(mi, mj, arith, a9, (a9 ? 94100 : 74100) + i, true));
    u32 result = 0;
    std::memcpy(&result, mj.host(BUF_BASE), sizeof result);
    if (result != arithmetic[i].output)
      std::fprintf(stderr, "thumb arith case %u ins=%04x input=%08x got=%08x expected=%08x\n",
                   i, arithmetic[i].instr, arithmetic[i].input,
                   result, arithmetic[i].output);
    CHECK(result == arithmetic[i].output);
  }
  unsetenv("DS_MIPS_NATIVE_BRANCH");
  unsetenv("DS_MIPS_NATIVE_MEM");
  jit::detach(mj.nds);
}

void native_user_bank_smoke(Cpu which) {
  const bool a9 = which == Cpu::ARM9;
  Machine mi(which), mj(which);
  setenv("DS_MIPS_NATIVE_OP", "all", 1);
  setenv("DS_MIPS_NATIVE_BRANCH", "1", 1);
  setenv("DS_MIPS_NATIVE_MEM", "1", 1);
  CHECK(jit::attach(mj.nds, a9, !a9));
  Trial t{};
  t.require_halt = true; t.budget = 1024;
  t.regs[0] = BUF_BASE; t.regs[1] = 0x12345000u;
  t.regs[2] = 0x87654321u; t.regs[13] = STACK; t.regs[14] = HALT_STUB;
  t.code = {0xE5801000, 0xE5802004, 0xE321F0D2, 0xE8D03000, 0xE8C03000,
            0xE321F0DF, 0xE580D008, 0xE580E00C};
  CHECK(run_both(mi, mj, t, a9, a9 ? 91147 : 71147, true));
  u32 values[4]{};
  std::memcpy(values, mj.host(BUF_BASE), sizeof values);
  CHECK(values[0] == 0x12345000u && values[1] == 0x87654321u &&
        values[2] == 0x87654321u && values[3] == HALT_STUB);
  t = Trial{};
  t.require_halt = true; t.budget = 1024;
  t.regs[1] = BUF_BASE; t.regs[6] = BUF_BASE + 0x100;
  t.regs[13] = STACK;
  t.code = {0xE321F0D2, 0xE8F10003, 0xE321F0DF, 0xE5860000, 0xE5861004};
  CHECK(run_both(mi, mj, t, a9, a9 ? 91148 : 71148, true));
  std::memcpy(values, mj.host(BUF_BASE + 0x100), 2 * sizeof(u32));
  CHECK(values[0] == 0xa5a5a5a5u && values[1] == 0xa4a4a4a4u);
  jit::detach(mj.nds);
}
#endif

void fuzz(Cpu which, bool thumb, u32 trials, u32 seed0) {
  const bool a9 = which == Cpu::ARM9;
  Machine mi(which), mj(which);
  CHECK(jit::attach(mj.nds, a9, !a9));
  u32 fails = 0;
  for (u32 n = 0; n < trials; ++n) {
    const u32 seed = seed0 + n;
    Rng r(seed);
    Trial t;
    t.thumb = thumb;
    const u32 len = 1 + r.below(24);
    for (u32 i = 0; i < len; ++i) {
      if (!thumb && r.coin(12)) { gen_user_bank(r, t.code); continue; }
      t.code.push_back(thumb ? gen_thumb(r) : gen_arm(r));
    }
    for (int i = 0; i < 15; ++i) t.regs[i] = r.coin(30) ? (r.below(5) - 2) : r.next();
    t.regs[9] = BUF_BASE + (r.below(4) << 10);
    t.regs[6] = BUF_BASE + (r.below(4) << 10);
    t.regs[13] = STACK;
    t.cpsr = (r.next() & 0xF0000000u);
    if (run_both(mi, mj, t, a9, seed, false)) continue;
    // Shrink: the shortest failing prefix names the instruction.
    Trial p = t;
    for (u32 k = 1; k <= t.code.size(); ++k) {
      p.code.assign(t.code.begin(), t.code.begin() + k);
      if (!run_both(mi, mj, p, a9, seed, false)) break;
    }
    run_both(mi, mj, p, a9, seed, true);
    if (++fails >= 3) break;
  }
  CHECK(fails == 0);
  std::printf("jit fuzz %s %s: %u trials ok (%u inconclusive, %u self-modifying)\n",
              a9 ? "arm9" : "arm7", thumb ? "thumb" : "arm", trials, g_inconclusive, g_selfmod);
  g_inconclusive = 0; g_selfmod = 0;
  jit::detach(mj.nds);
}

} // namespace

int main(int argc, char** argv) {
#if defined(__mips__)
  if (argc > 1 && (std::strcmp(argv[1], "nativebank9") == 0 ||
                   std::strcmp(argv[1], "nativebank7") == 0)) {
    native_user_bank_smoke(argv[1][10] == '9' ? Cpu::ARM9 : Cpu::ARM7);
    return 0;
  }
  if (argc > 1 && (std::strcmp(argv[1], "thumbsmoke9") == 0 ||
                   std::strcmp(argv[1], "thumbsmoke7") == 0)) {
    native_thumb_smoke(argv[1][10] == '9' ? Cpu::ARM9 : Cpu::ARM7);
    return 0;
  }
  if (argc > 1 && (std::strcmp(argv[1], "nativepc9") == 0 ||
                   std::strcmp(argv[1], "nativepc7") == 0)) {
    const bool a9 = argv[1][8] == '9';
    setenv("DS_MIPS_NATIVE", "1", 1);
    setenv("DS_MIPS_NATIVE_OP", "all", 1);
    setenv("DS_MIPS_NATIVE_ARM7", "1", 1);
    setenv("DS_MIPS_PC_ONLY", "1", 1);
    native_compare_loop(a9 ? Cpu::ARM9 : Cpu::ARM7);
    return 0;
  }
#endif
  if (argc > 3 && std::strcmp(argv[1], "directed") == 0) {
    directed(static_cast<u32>(std::atoi(argv[2])), static_cast<u32>(std::atoi(argv[3])));
    return 0;
  }
  if (argc > 1 && std::strncmp(argv[1], "native", 6) == 0) {
    native_alu(argv[1][6] == '9' ? 0 : argv[1][6] == '7' ? 1 : -1);
    return 0;
  }
  const u32 trials = argc > 1 ? static_cast<u32>(std::atoi(argv[1])) : 400;
  // test_jit <count> <seed> [set]: run <count> trials from <seed> in one set,
  // and nothing else. <count> 1 is the single-trial form; a larger count
  // reproduces a failure that only appears with the set's earlier trials run
  // before it (the runtime is global and carries state across them).
  if (argc > 2) {   // set = 0..3 (arm9 arm/thumb, arm7 arm/thumb), default by seed range
    const u32 seed = static_cast<u32>(std::atoi(argv[2]));
    const int set = argc > 3 ? std::atoi(argv[3]) : (seed >= 3000 ? 2 : 0) + ((seed / 1000) % 2 == 0 ? 1 : 0);
    fuzz(set >= 2 ? Cpu::ARM7 : Cpu::ARM9, set & 1, trials, seed);
    return 0;
  }
  directed();
  native_alu();
  fuzz(Cpu::ARM9, false, trials, 1000);
  fuzz(Cpu::ARM9, true, trials, 2000);
  fuzz(Cpu::ARM7, false, trials, 3000);
  fuzz(Cpu::ARM7, true, trials, 4000);
  const jit::Stats& s = jit::stats();
  std::printf("jit: %llu blocks, %llu inline instrs, %llu fallbacks\n",
              (unsigned long long)s.blocks_translated, (unsigned long long)s.instrs_translated, (unsigned long long)s.instrs_fallback);
  std::puts("jit: ok");
  return 0;
}
