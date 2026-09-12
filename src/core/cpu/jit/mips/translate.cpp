#include "core/cpu/jit/jit_internal.h"
#include "core/cpu/jit/mips/emit.h"
namespace ds::jit::backend {
// Fallback-first milestone: runtime keeps using the interpreter until the
// MIPS block ABI is wired into dispatch. This deliberately emits no block.
bool translate_block(JitCpu&,u32,u8*,size_t,Block&,u32& size){size=0;return false;}
}
