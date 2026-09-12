#include "core/cpu/jit/jit_internal.h"
#include "core/cpu/jit/mips/emit.h"
#include <cstring>
#include <vector>
extern "C" ds::u32 ds_jit_mips_fallback(ds::CpuContext*, ds::u32, ds::u32);
namespace ds::jit::backend {
bool translate_block(JitCpu& jc, u32 key, u8* buf, size_t cap, Block& b, u32& size) {
  const u32 start = key_pc(key), step = key_thumb(key) ? 2 : 4; u8* host = jc.ctx->page_table.read_ptr(start); if (!host || cap < 64) return false;
  MipsEmitter e(buf, cap); e.addiu(29, 29, -8); e.sw(31, 4, 29); std::vector<size_t> exits; u32 count = 0, addr = start;
  for (; count < 8; ++count, addr += step) {
    u8* p = jc.ctx->page_table.read_ptr(addr); if (!p) break; u32 instr = 0; std::memcpy(&instr, p, step);
    e.lui(5, instr >> 16); e.ori(5, 5, instr); const u32 k = make_key(addr, key_thumb(key));
    e.lui(6, k >> 16); e.ori(6, 6, k); e.load_ptr(25, reinterpret_cast<const void*>(&ds_jit_mips_fallback)); e.jalr(31, 25); e.nop();
    exits.push_back(e.bnez(2)); e.nop();
    const bool arm_branch = !key_thumb(key) && (((instr & 0x0e000000u) == 0x0a000000u) || ((instr & 0x0ffffff0u) == 0x012fff10u));
    const bool thumb_branch = key_thumb(key) && (((instr & 0xf000u) == 0xd000u) || ((instr & 0xf800u) == 0xe000u));
    if (arm_branch || thumb_branch) break;
  }
  if (!count) return false; const size_t done = e.size(); e.lw(31, 4, 29); e.addiu(29, 29, 8); e.jr(31); e.nop();
  for (size_t p : exits) e.patch_branch(p, done); size = static_cast<u32>(e.size()); const u32 len = addr - start;
  b.guest_len = len; b.hot_size = size; b.npages = 1; b.nsucc = 0; b.dead = false;
  b.host_pages[0] = reinterpret_cast<const u8*>(reinterpret_cast<uintptr_t>(host) & ~uintptr_t{4095}); b.host_lo = host; b.host_hi = host + len - 1; return true;
}
}
extern "C" ds::u32 ds_jit_mips_fallback(ds::CpuContext* c, ds::u32 instr, ds::u32 key) { return ds::jit::jit_h_fallback(c, instr, key); }
