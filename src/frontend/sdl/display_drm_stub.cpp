// SPDX-License-Identifier: GPL-3.0-or-later
#include "display_drm.h"

namespace ds::sdl {
bool DrmOut::open(SDL_Window*, int, int, int) { return false; }
void DrmOut::close() {}
u32* DrmOut::begin_frame() { return nullptr; }
void DrmOut::end_frame() {}
void DrmOut::flush() {}
bool DrmOut::flip(int) { return false; }
bool DrmOut::alloc_buf(Buf&) { return false; }
void DrmOut::drop_buf(Buf&) {}
bool DrmOut::pump(int, bool) { return false; }
void DrmOut::retire() {}
} // namespace ds::sdl
