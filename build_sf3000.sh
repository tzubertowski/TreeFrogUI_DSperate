#!/bin/sh
set -eu

cd "$(dirname "$0")"
device_sdl=${SF3000_SDL2:-../SF3000_sdcard/SF3000_sdcard/cubegm/usr/lib/libSDL2.so}
test -f "$device_sdl"

docker build --network host -t dsperate-sf3000-build -f cmake/sf3000.Dockerfile .
docker run --rm -v "$PWD:/src" dsperate-sf3000-build rm -rf /src/build/sf3000 /src/build/sf3000-package
docker run --rm --user "$(id -u):$(id -g)" --network host -v "$PWD:/src" -v "$(realpath "$device_sdl"):/device/libSDL2.so:ro" dsperate-sf3000-build sh -c '
  set -eu
  PKG_CONFIG_LIBDIR=/opt/sdl/usr/lib/mipsel-linux-gnu/pkgconfig \
  PKG_CONFIG_SYSROOT_DIR=/opt/sdl cmake -S . -B build/sf3000 -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_PROCESSOR=mipsel \
    -DCMAKE_C_COMPILER=mipsel-linux-gnu-gcc -DCMAKE_CXX_COMPILER=mipsel-linux-gnu-g++ \
    -DCMAKE_C_FLAGS="-mips32r2 -mtune=74kc -mdspr2 -mhard-float" \
    -DCMAKE_CXX_FLAGS="-mips32r2 -mtune=74kc -mdspr2 -mhard-float -I/opt/sdl/usr/include -I/opt/sdl/usr/include/mipsel-linux-gnu" \
    -DCMAKE_EXE_LINKER_FLAGS="-static-libstdc++ -static-libgcc -L/device" \
    -DSDL2_INCLUDE_DIRS=/opt/sdl/usr/include/SDL2 \
    -DDSPERATE_JIT=ON -DDSPERATE_MIPS_JIT=ON -DDSPERATE_NEON=OFF -DDSPERATE_TESTS=OFF \
    -DDSPERATE_WAYLAND=OFF -DDSPERATE_CHEEVOS=OFF -DDSPERATE_HEADLESS=OFF
  PKG_CONFIG_LIBDIR=/opt/sdl/usr/lib/mipsel-linux-gnu/pkgconfig \
    PKG_CONFIG_SYSROOT_DIR=/opt/sdl cmake --build build/sf3000 -j"$(nproc)"
  rm -rf build/sf3000-package
  mkdir -p build/sf3000-package/lib
  cp build/sf3000/src/frontend/sdl/dsperate build/sf3000-package/
  cp run_sf3000.sh build/sf3000-package/
  cp /device/libSDL2.so build/sf3000-package/lib/libSDL2-2.0.so.0
  cp /usr/mipsel-linux-gnu/lib/ld.so.1 /usr/mipsel-linux-gnu/lib/libc.so.6 \
     /usr/mipsel-linux-gnu/lib/libm.so.6 /usr/mipsel-linux-gnu/lib/libdl.so.2 \
     /usr/mipsel-linux-gnu/lib/libpthread.so.0 /usr/mipsel-linux-gnu/lib/librt.so.1 \
     /usr/mipsel-linux-gnu/lib/libatomic.so.1 build/sf3000-package/lib/
'
