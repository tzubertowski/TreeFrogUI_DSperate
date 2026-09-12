#include "core/cpu/jit/jit_internal.h"
#include "core/cpu/jit/mips/emit.h"
#include <cstring>
extern "C" ds::u32 ds_jit_mips_fallback(ds::CpuContext*, ds::u32, ds::u32);
namespace ds::jit::backend {
bool translate_block(JitCpu& jc, u32 key, u8* buf, size_t cap, Block& b, u32& size) {
  const u32 pc = key_pc(key), len = key_thumb(key) ? 2 : 4;
  u8* host = jc.ctx->page_table.read_ptr(pc); if (!host || cap < 40) return false;
  u32 instr = 0; std::memcpy(&instr, host, len); MipsEmitter e(buf, cap);
  e.addiu(29, 29, -8); e.sw(31, 4, 29); e.lui(5, instr >> 16); e.ori(5, 5, instr);
  e.lui(6, key >> 16); e.ori(6, 6, key); e.load_ptr(25, reinterpret_cast<const void*>(&ds_jit_mips_fallback));
  e.jalr(31, 25); e.nop(); e.lw(31, 4, 29); e.addiu(29, 29, 8); e.jr(31); e.nop();
  size = static_cast<u32>(e.size()); b.guest_len = len; b.hot_size = size; b.npages = 1; b.nsucc = 0; b.dead = false;
  b.host_pages[0] = reinterpret_cast<const u8*>(reinterpret_cast<uintptr_t>(host) & ~uintptr_t{4095}); b.host_lo = host; b.host_hi = host + len - 1; return true;
}
}
extern "C" ds::u32 ds_jit_mips_fallback(ds::CpuContext* c, ds::u32 instr, ds::u32 key) { return ds::jit::jit_h_fallback(c, instr, key); }
