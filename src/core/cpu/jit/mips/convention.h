#pragma once
#include "core/cpu/jit/jit_internal.h"

namespace ds::jit {
// MIPS o32: a0-a3 are helper arguments, s0 is CpuContext*, t9 is call target.
// Entry redirects carry their key in t0's delay-slot-built value; dispatch
// moves it to a1 before calling jit_h_lookup(cpu, key).
constexpr u32 R_CTX=16, R_FN=25, R_RA=31, R_SP=29;
constexpr u32 SCRATCH0=8, SCRATCH1=9, SCRATCH2=10, SCRATCH3=11;
}
