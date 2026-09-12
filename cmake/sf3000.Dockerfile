FROM debian:bookworm
RUN dpkg --add-architecture mipsel \
 && apt-get update \
 && apt-get install -y --no-install-recommends cmake ninja-build gcc-mipsel-linux-gnu g++-mipsel-linux-gnu pkg-config \
 && cd /tmp \
 && apt-get download libsdl2-dev:mipsel \
 && mkdir /opt/sdl \
 && dpkg-deb -x libsdl2-dev_*_mipsel.deb /opt/sdl \
 && rm -rf /var/lib/apt/lists/* /tmp/libsdl2-dev_*
WORKDIR /src
