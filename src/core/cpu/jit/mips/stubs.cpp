#include "core/cpu/jit/jit_internal.h"
#include "core/cpu/jit/mips/convention.h"
#include "core/cpu/jit/mips/emit.h"
#include "core/sched/scheduler.h"
namespace ds { extern "C" SliceNext ds_slice_next(void*); }
extern "C" const void* jit_h_lookup(ds::CpuContext*, ds::u32);
namespace ds::jit::backend {
using namespace ds::jit;
extern "C" void ds_jit_mips_run_loop(void* scheduler) {
  for (;;) {
    ds::SliceNext next = ds::ds_slice_next(scheduler);
    if (!next.ctx) return;
    ds::jit::rt().enter(next.ctx, next.native);
  }
}
void emit_stubs(Runtime& rt) {
  MipsEmitter e(rt.arena + LUT_AREA, rt.cap - LUT_AREA);
  rt.enter = reinterpret_cast<void (*)(CpuContext*, const void*)>(e.cur());
  // a0=CpuContext, a1=native. Tail-jump so the native block returns to our
  // caller using its own saved ra.
  e.jr(5); e.nop();
  rt.enter_light=e.cur(); rt.run_loop=&ds_jit_mips_run_loop; rt.exit_key=e.cur(); rt.exit_key_lit=nullptr;
  rt.exit_r15=nullptr; rt.call_pure=nullptr; rt.call_full=nullptr; rt.call2=nullptr; rt.poll=nullptr;
  // Arena exhaustion returns here from the dispatcher.  A null target used
  // to turn a harmless refill into a jump to address zero on long runs.
  rt.flush_exit=e.cur();
  e.jr(R_RA); e.nop();
  for(auto& c:rt.cpus){
    c.dispatch=e.cur();
    // Keep the 16-byte o32 outgoing argument area and 8-byte stack alignment.
    e.addiu(R_SP,R_SP,-24); e.sw(R_RA,20,R_SP); e.sw(4,16,R_SP);
    e.ori(5,SCRATCH0,0); // redirect delay slot completed t0=key
    e.load_ptr(R_FN, reinterpret_cast<const void*>(&jit_h_lookup));
    e.jalr(R_RA,R_FN); e.nop();
    e.lw(4,16,R_SP); e.lw(R_RA,20,R_SP); e.addiu(R_SP,R_SP,24);
    e.jr(2); e.nop();
    c.link=nullptr;c.fallback=nullptr;c.branch_indirect=nullptr;c.branch_indirect_cdi=nullptr;
  }
  rt.stubs_end=LUT_AREA+e.size(); rt.pos=rt.stubs_end;
  // The first JIT entry is executed immediately after attach.  MIPS has
  // separate I/D caches; make the freshly emitted dispatcher visible before
  // handing its address to the runtime.
  sync_icache(rt.arena + LUT_AREA, e.size());
}
void write_entry_redirect(u8* entry,u32 key,const u8* dispatch){MipsEmitter e(entry,ENTRY_PATCH);e.lui(SCRATCH0,key>>16);e.j(reinterpret_cast<uintptr_t>(dispatch));e.ori(SCRATCH0,SCRATCH0,key);}
void patch_link(u8* site,const u8* target){MipsEmitter::patch(site,0x08000000u|((reinterpret_cast<uintptr_t>(target)>>2)&0x03ffffff));}
u32 relative_branch_class(u32 w){return (w&0xfc000000u)==0x08000000u?1:0;}
}
