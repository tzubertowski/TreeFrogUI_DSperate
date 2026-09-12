#!/bin/sh
set -eu

sd="/run/media/$USER/R36HD/cubegm/dsperate"
repo="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
[ -d "$sd" ] || { echo "SD path not found: $sd" >&2; exit 1; }

cp "$repo/build/sf3000-package/dsperate" "$sd/dsperate"
cp "$repo/run_sf3000.sh" "$sd/run_sf3000.sh"
sync
echo "DSperate copied to $sd"
