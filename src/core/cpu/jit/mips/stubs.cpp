#include "core/cpu/jit/jit_internal.h"
#include "core/cpu/jit/mips/convention.h"
#include "core/cpu/jit/mips/emit.h"
namespace ds::jit::backend {
using namespace ds::jit;
void emit_stubs(Runtime& rt) {
  MipsEmitter e(rt.arena + LUT_AREA, rt.cap - LUT_AREA);
  rt.enter = reinterpret_cast<void (*)(CpuContext*, const void*)>(e.cur());
  // a0=CpuContext, a1=native. Keep the ABI frame-free for this first tier.
  e.jalr(31,5); e.nop(); e.jr(31); e.nop();
  rt.enter_light=e.cur(); rt.run_loop=nullptr; rt.exit_key=e.cur(); rt.exit_key_lit=nullptr;
  rt.exit_r15=nullptr; rt.call_pure=nullptr; rt.call_full=nullptr; rt.call2=nullptr; rt.poll=nullptr; rt.flush_exit=nullptr;
  rt.stubs_end=LUT_AREA+e.size(); rt.pos=rt.stubs_end;
  for(auto& c:rt.cpus){c.dispatch=nullptr;c.link=nullptr;c.fallback=nullptr;c.branch_indirect=nullptr;c.branch_indirect_cdi=nullptr;}
}
void write_entry_redirect(u8* entry,u32 key,const u8* dispatch){MipsEmitter e(entry,ENTRY_PATCH);e.lui(4,key>>16);e.ori(4,4,key);e.j(reinterpret_cast<uintptr_t>(dispatch));}
void patch_link(u8* site,const u8* target){MipsEmitter::patch(site,0x08000000u|((reinterpret_cast<uintptr_t>(target)>>2)&0x03ffffff));}
u32 relative_branch_class(u32 w){return (w&0xfc000000u)==0x08000000u?1:0;}
}
