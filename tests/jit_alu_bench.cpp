// SPDX-License-Identifier: GPL-3.0-or-later
// Small ARM ALU A/B benchmark. Run the same binary with DS_MIPS_NATIVE=0/1.
#include "core/nds.h"
#include "core/cpu/interp/interp.h"
#include "core/cpu/jit/jit.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace ds;

namespace {
constexpr u32 CODE = 0x02000000, HALT = 0x02001000, STACK = 0x02300000;

struct Machine {
  NDS nds;
  CpuContext& cpu;
  explicit Machine(Cpu which) : cpu(nds.cpu(which)) {}
  u8* host(u32 a) { return cpu.page_table.read_ptr(a); }
  void poke32(u32 a, u32 v) { std::memcpy(host(a), &v, 4); }
};

void halt_stub(Machine& m, bool arm9) {
  for (u32 v = 0; v < 0x20; v += 4) m.poke32(v, 0xE59FF000u | (0x20 - v - 8));
  m.poke32(0x20, HALT);
  if (arm9) {
    m.poke32(HALT, 0xEE070F90); // mcr p15, 0, r0, c7, c0, 4
  } else {
    m.poke32(HALT, 0xE3A00301); // mov r0, #0x04000000
    m.poke32(HALT + 4, 0xE2800C03);
    m.poke32(HALT + 8, 0xE3A01080);
    m.poke32(HALT + 12, 0xE5C01001); // strb r1, [r0, #0x301]
  }
  m.poke32(HALT + (arm9 ? 4 : 16), 0xEAFFFFFE);
}

u32 imm(u32 op, u32 rd, u32 rn, u32 rot, u32 value, bool setflags = false, u32 cond = 14) {
  return (cond << 28) | (1u << 25) | (setflags ? (1u << 20) : 0) | (op << 21) | (rn << 16) | (rd << 12) |
         (rot << 8) | value;
}
u32 reg(u32 op, u32 rd, u32 rn, u32 rm, u32 shift = 0, u32 type = 0) {
  return 0xE0000000u | (op << 21) | (rn << 16) | (rd << 12) | (shift << 7) | (type << 5) | rm;
}

std::vector<u32> workload() {
  const u32 ops[] = {13, 15, 4, 2, 0, 1, 12, 14};
  const u32 rots[] = {0, 1, 4, 8, 12, 15};
  std::vector<u32> code;
  code.reserve(192);
  for (u32 repeat = 0; repeat < 8; ++repeat) {
    for (u32 i = 0; i < 8; ++i) {
      const u32 op = ops[i];
      code.push_back(imm(op, i & 7, (i + 1) & 7, rots[(i + repeat) % 6], 0x11u + i * 0x13u));
      const u32 type = (i + repeat) % 3;
      const u32 shift = type ? ((i + repeat) % 31) + 1 : (i + repeat) & 31;
      code.push_back(reg(op, (i + 2) & 7, i & 7, (i + 3) & 7, shift, type));
    }
    code.push_back(imm(13, 7, 0, repeat % 16, 0x80u + repeat, true));
    code.push_back(imm(13, 7, 0, 0, 0, true));
    code.push_back(imm(13, 6, 0, 0, 0x66, false, 0));
    code.push_back(imm(13, 6, 0, 0, 0x99, false, 1));
  }
  return code;
}

void load(Machine& m, const std::vector<u32>& code, bool arm9, bool install = true) {
  if (install) {
    halt_stub(m, arm9);
    for (size_t i = 0; i < code.size(); ++i) m.poke32(CODE + static_cast<u32>(i * 4), code[i]);
    const u32 at = CODE + static_cast<u32>(code.size() * 4);
    m.poke32(at, 0xEA000000u | ((HALT - (at + 8)) >> 2));
  }
  auto& c = m.cpu;
  c.set_cpsr(0x1F); for (int i = 0; i < 15; ++i) c.hot.regs[i] = 0x10203040u + i * 0x11111111u;
  c.hot.regs[13] = STACK; c.hot.cpsr = 0x2000001F; c.hot.regs[15] = CODE + 8;
  c.halted = false; c.hot.irq_pending = 0; c.code_cycles = CODE >> 15; c.code_region = CODE >> 24;
  c.hot.cycle_budget = 1 << 24; c.budget_at_halt = 0; c.jumped = false;
}

template <typename Run>
void run(Machine& m, Run fn) {
  for (int guard = 0; guard < 100000 && !m.cpu.halted && m.cpu.hot.cycle_budget > 0; ++guard) fn(m.cpu);
}

bool same(const Machine& a, const Machine& b) {
  if (!a.cpu.halted || !b.cpu.halted || a.cpu.budget_at_halt != b.cpu.budget_at_halt ||
      a.cpu.hot.cpsr != b.cpu.hot.cpsr) {
    std::fprintf(stderr, "jit ALU state mismatch: halted=%d/%d budget=%d/%d cpsr=%08x/%08x\n",
                 a.cpu.halted, b.cpu.halted, a.cpu.budget_at_halt, b.cpu.budget_at_halt,
                 a.cpu.hot.cpsr, b.cpu.hot.cpsr);
    return false;
  }
  for (int i = 0; i < 16; ++i) if (a.cpu.hot.regs[i] != b.cpu.hot.regs[i]) {
    std::fprintf(stderr, "jit ALU mismatch r%d: interp=%08x jit=%08x\n", i,
                 a.cpu.hot.regs[i], b.cpu.hot.regs[i]);
    return false;
  }
  return true;
}
}

int main() {
  const bool arm9 = std::getenv("DS_MIPS_BENCH_ARM7") == nullptr;
  const std::vector<u32> code = workload();
  Machine mi(arm9 ? Cpu::ARM9 : Cpu::ARM7), mj(arm9 ? Cpu::ARM9 : Cpu::ARM7);
  if (!jit::attach(mj.nds, arm9, !arm9)) return 2;

  load(mi, code, arm9); load(mj, code, arm9); run(mi, &interp::run); run(mj, &jit::run);
  if (!same(mi, mj)) {
    for (size_t i = 0; i < code.size(); ++i)
      std::fprintf(stderr, "%08x: %08x\n", CODE + static_cast<u32>(i * 4), code[i]);
    std::fprintf(stderr, "jit ALU benchmark mismatch\n");
    return 1;
  }

  constexpr int samples = 24;
  auto measure = [&](Machine& m, auto fn) {
    const auto begin = std::chrono::steady_clock::now();
    for (int i = 0; i < samples; ++i) { load(m, code, arm9, false); run(m, fn); }
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
  };
  const double interp_ms = measure(mi, &interp::run);
  const double jit_ms = measure(mj, &jit::run);
  std::printf("jit ALU benchmark: cpu=ARM%d native=%s samples=%d instructions=%zu interp_ms=%.3f jit_ms=%.3f\n",
              arm9 ? 9 : 7,
              std::getenv("DS_MIPS_NATIVE") ? std::getenv("DS_MIPS_NATIVE") : "default",
              samples, code.size(), interp_ms, jit_ms);
  jit::detach(mj.nds);
  return 0;
}
