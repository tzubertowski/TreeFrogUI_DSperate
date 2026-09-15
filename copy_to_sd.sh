#!/bin/sh
set -eu

repo="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
sd=""
for root in "/run/media/$USER/R36HD" "$HOME/SDCARD"; do
  if [ -d "$root/cubegm/dsperate" ]; then sd="$root/cubegm/dsperate"; break; fi
done
[ -n "$sd" ] || { echo "SD card not found" >&2; exit 1; }

src="$repo/build/sf3000-package/dsperate"
cp "$src" "$sd/dsperate"
cp "$repo/run_sf3000.sh" "$sd/run_sf3000.sh"
sync
src_hash=$(sha256sum "$src" | awk '{print $1}')
dst_hash=$(sha256sum "$sd/dsperate" | awk '{print $1}')
[ "$src_hash" = "$dst_hash" ] || {
  echo "DSperate verification failed: source=$src_hash sd=$dst_hash" >&2
  exit 1
}
echo "DSperate copied and verified: $dst_hash"
