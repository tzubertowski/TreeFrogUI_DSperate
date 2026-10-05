// SPDX-License-Identifier: GPL-3.0-or-later
#include "display_drm.h"
#include <SDL2/SDL.h>
#include <cstring>

namespace ds::sdl {
bool DrmOut::open(SDL_Window*, int, int, int) { return false; }
bool DrmOut::dmabuf_plane(int, DmabufPlane&) const { return false; }
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

// The SF3000 SDL2 library predates these optional query symbols, while the
// upstream headers expose them. Keep the newer frontend source linkable.
extern "C" int SDL_GetDefaultAudioInfo(char **name, SDL_AudioSpec *spec, int) {
  if (name) *name = nullptr;
  if (spec) std::memset(spec, 0, sizeof(*spec));
  return -1;
}
extern "C" const char *SDL_GetTouchName(int) { return nullptr; }
extern "C" const char *SDL_JoystickPath(SDL_Joystick *) { return nullptr; }
