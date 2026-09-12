// SPDX-License-Identifier: GPL-3.0-or-later
#include "display_hcge.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <dlfcn.h>
#include <fcntl.h>
#include <unistd.h>

namespace ds::sdl {
namespace {
const char* driver_path() {
  static char path[128];
  if (path[0]) return path;
  const int fd = ::open("/tmp/tfdevice.env", O_RDONLY);
  if (fd >= 0) {
    char buf[512]; const ssize_t n = ::read(fd, buf, sizeof(buf) - 1); ::close(fd);
    if (n > 0) {
      buf[n] = 0; char* p = std::strstr(buf, "TF_DRIVER=");
      while (p && p != buf && p[-1] != '\n') p = std::strstr(p + 1, "TF_DRIVER=");
      if (p) {
        p += 10; char* e = p; while (*e && *e != '\r' && *e != '\n') ++e;
        const size_t len = static_cast<size_t>(e - p);
        if (len && len < sizeof(path)) { std::memcpy(path, p, len); path[len] = 0; }
      }
    }
  }
  if (!path[0]) std::strcpy(path, "/mnt/sdcard/cubegm/driver.so");
  return path;
}
} // namespace

u16 hcge_rgb565(u32 p) { return static_cast<u16>(((p >> 8) & 0xF800) | ((p >> 5) & 0x07E0) | ((p >> 3) & 0x001F)); }

bool HcgeOut::open() {
  std::fprintf(stderr, "hcge: loading %s\n", driver_path()); std::fflush(stderr);
  handle_ = ::dlopen(driver_path(), RTLD_NOW | RTLD_GLOBAL);
  if (!handle_) { std::fprintf(stderr, "hcge: %s\n", ::dlerror()); return false; }
  using Init = int (*)();
  auto init = reinterpret_cast<Init>(::dlsym(handle_, "video_drivers_init"));
  deinit_ = reinterpret_cast<Deinit>(::dlsym(handle_, "video_driver_deinit"));
  present_ = reinterpret_cast<Present>(::dlsym(handle_, "video_driver_disp_frame"));
  std::fprintf(stderr, "hcge: init begin\n"); std::fflush(stderr);
  if (!init || !deinit_ || !present_ || init() <= 0) { close(); return false; }
  std::fprintf(stderr, "video: SF3000 HCGE scaler\n");
  return true;
}

void HcgeOut::close() {
  if (handle_ && deinit_) deinit_();
  if (handle_) ::dlclose(handle_);
  handle_ = nullptr; deinit_ = nullptr; present_ = nullptr;
  frame_[0].clear(); frame_[1].clear();
}

void HcgeOut::present(const u32* const fb[2], const int screen[2], const int rect[2][4],
                      const bool shown[2], u8 inset_alpha, int w, int h) {
  if (!present_ || w <= 0 || h <= 0) return;
  const int out_w = 320, out_h = 240;
  auto& dst = frame_[page_++ & 1];
  dst.assign(static_cast<size_t>(out_w) * out_h, 0);
  for (int i = 0; i < 2; ++i) {
    if (!shown[i]) continue;
    const int* r = rect[i];
    for (int y = 0; y < r[3]; ++y) {
      const int sy = y * static_cast<int>(SCREEN_H) / r[3];
      for (int x = 0; x < r[2]; ++x) {
        const int dx = (r[0] + x) * out_w / w, dy = (r[1] + y) * out_h / h;
        if (dx < 0 || dx >= out_w || dy < 0 || dy >= out_h) continue;
        const u16 src = hcge_rgb565(fb[screen[i]][static_cast<size_t>(sy) * SCREEN_W + x * static_cast<int>(SCREEN_W) / r[2]]);
        u16& out = dst[static_cast<size_t>(dy) * out_w + dx];
        if (i && inset_alpha < 255) {
          const unsigned a = inset_alpha + (inset_alpha >> 7), b = 256 - a;
          const unsigned r5 = (((src >> 11) * a + (out >> 11) * b) >> 8) & 31;
          const unsigned g6 = ((((src >> 5) & 63) * a + ((out >> 5) & 63) * b) >> 8) & 63;
          const unsigned b5 = (((src & 31) * a + (out & 31) * b) >> 8) & 31;
          out = static_cast<u16>((r5 << 11) | (g6 << 5) | b5);
        } else out = src;
      }
    }
  }
  if (std::getenv("DS_HCGE_DIAG")) {
    static unsigned frames;
    static timespec started;
    if (!frames) ::clock_gettime(CLOCK_MONOTONIC, &started);
    if (!(frames % 10)) {
      timespec now; ::clock_gettime(CLOCK_MONOTONIC, &now);
      const double seconds = now.tv_sec - started.tv_sec + (now.tv_nsec - started.tv_nsec) / 1e9;
      u32 hash = 2166136261u;
      for (size_t p = 0; p < dst.size(); p += 257) hash = (hash ^ dst[p]) * 16777619u;
      std::fprintf(stderr, "hcge: frame=%u %dx%d->%dx%d elapsed=%.3f fps=%.2f hash=%08x\n",
                   frames + 1, w, h, out_w, out_h, seconds, frames ? frames / seconds : 0.0, hash);
      std::fflush(stderr);
    }
    ++frames;
  }
  present_(dst.data(), out_w, out_h, out_w * static_cast<int>(sizeof(u16)));
}
} // namespace ds::sdl
