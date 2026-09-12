#include "frontend/sdl/display_hcge.h"
#include "check.h"

int main() {
  using ds::sdl::hcge_rgb565;
  CHECK(hcge_rgb565(0xFFFF0000) == 0xF800);
  CHECK(hcge_rgb565(0xFF00FF00) == 0x07E0);
  CHECK(hcge_rgb565(0xFF0000FF) == 0x001F);
  CHECK(hcge_rgb565(0xFFFFFFFF) == 0xFFFF);
  return 0;
}
