#include "core/cpu/jit/jit_internal.h"
#include "core/cpu/jit/mips/emit.h"
#include <cstring>
#include <vector>
extern "C" ds::u32 ds_jit_mips_fallback(ds::CpuContext*, ds::u32, ds::u32);
extern "C" ds::u32 ds_jit_mips_fallback_block(ds::CpuContext*, const ds::u32*, const ds::u32*, ds::u32);
namespace ds::jit::backend {
bool translate_block(JitCpu& jc, u32 key, u8* buf, size_t cap, Block& b, u32& size) {
  const u32 start = key_pc(key), step = key_thumb(key) ? 2 : 4; u8* host = jc.ctx->page_table.read_ptr(start); if (!host || cap < 64) return false;
  MipsEmitter e(buf, cap); e.addiu(29, 29, -8); e.sw(31, 4, 29); std::vector<u32> instrs, keys; u32 count = 0, addr = start;
  for (; count < 32; ++count, addr += step) {
    u8* p = jc.ctx->page_table.read_ptr(addr); if (!p) break; u32 instr = 0; std::memcpy(&instr, p, step);
    instrs.push_back(instr); keys.push_back(make_key(addr, key_thumb(key)));
    const bool arm_branch = !key_thumb(key) && (((instr & 0x0e000000u) == 0x0a000000u) || ((instr & 0x0ffffff0u) == 0x012fff10u));
    const bool thumb_branch = key_thumb(key) && (((instr & 0xf000u) == 0xd000u) || ((instr & 0xf800u) == 0xe000u));
    if (arm_branch || thumb_branch) break;
  }
  if (!count) return false;
  const size_t ip = e.size(); e.lui(5, 0); e.ori(5, 5, 0); const size_t kp = e.size(); e.lui(6, 0); e.ori(6, 6, 0);
  e.addiu(7, 0, static_cast<s32>(count)); e.load_ptr(25, reinterpret_cast<const void*>(&ds_jit_mips_fallback_block)); e.jalr(31, 25); e.nop();
  e.lw(31, 4, 29); e.addiu(29, 29, 8); e.jr(31); e.nop();
  while (e.size() & 3) e.nop(); const size_t idata = e.size(); for (u32 v : instrs) e.w(v); const size_t kdata = e.size(); for (u32 v : keys) e.w(v);
  e.patch_ptr(ip, 5, buf + idata); e.patch_ptr(kp, 6, buf + kdata); size = static_cast<u32>(e.size()); const u32 len = addr - start;
  b.guest_len = len; b.hot_size = size; b.npages = 1; b.nsucc = 0; b.dead = false;
  b.host_pages[0] = reinterpret_cast<const u8*>(reinterpret_cast<uintptr_t>(host) & ~uintptr_t{4095}); b.host_lo = host; b.host_hi = host + len - 1; return true;
}
}
extern "C" ds::u32 ds_jit_mips_fallback(ds::CpuContext* c, ds::u32 instr, ds::u32 key) { return ds::jit::jit_h_fallback(c, instr, key); }
extern "C" ds::u32 ds_jit_mips_fallback_block(ds::CpuContext* c, const ds::u32* ins, const ds::u32* keys, ds::u32 n) {
  for (ds::u32 i = 0; i < n; ++i) if (ds::jit::jit_h_fallback(c, ins[i], keys[i])) return 1;
  return 0;
}
