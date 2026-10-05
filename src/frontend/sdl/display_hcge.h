#pragma once
#include "core/types.h"
#include <vector>
namespace ds::sdl {
class HcgeOut {
public:
  bool open(); void close();
  void present(const u32* const fb[2], const int screen[2], const int rect[2][4], const bool shown[2], u8 alpha, int w, int h);
  explicit operator bool() const { return handle_ != nullptr; }
private:
  using Deinit = void (*)(); using Present = int (*)(void*, int, int, int);
  void* handle_ = nullptr; Deinit deinit_ = nullptr; Present present_ = nullptr;
  std::vector<u16> frame_[2]; unsigned page_ = 0;
};
}
