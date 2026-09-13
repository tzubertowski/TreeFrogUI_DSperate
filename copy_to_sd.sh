#!/bin/sh
set -eu

repo="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
sd=""
for root in "/run/media/$USER/R36HD" "$HOME/SDCARD"; do
  if [ -d "$root/cubegm/dsperate" ]; then sd="$root/cubegm/dsperate"; break; fi
done
[ -n "$sd" ] || { echo "SD card not found" >&2; exit 1; }

cp "$repo/build/sf3000-package/dsperate" "$sd/dsperate"
cp "$repo/run_sf3000.sh" "$sd/run_sf3000.sh"
sync
echo "DSperate copied to $sd"
