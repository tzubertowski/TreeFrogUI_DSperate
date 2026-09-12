#!/bin/sh
set -eu

sd="/run/media/$USER/R36HD/cubegm/dsperate"
[ -d "$sd" ] || { echo "SD path not found: $sd" >&2; exit 1; }

cp build/sf3000-package/dsperate "$sd/dsperate"
cp run_sf3000.sh "$sd/run_sf3000.sh"
sync
echo "DSperate copied to $sd"
