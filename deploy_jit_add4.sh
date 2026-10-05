#!/bin/sh
set -eu

repo=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
limit=${1:-4}
card=/run/media/${USER}/R36HD
dst=$card/cubegm/dsperate
[ -d "$dst" ] || { echo "R36HD not mounted at $card" >&2; exit 1; }
[ -w "$dst" ] || { echo "R36HD is read-only; remount it read-write" >&2; exit 1; }

src=$repo/build/sf3000-package/dsperate
[ -f "$src" ] || { echo "build first: $src missing" >&2; exit 1; }
launcher=$(mktemp)
trap 'rm -f "$launcher"' EXIT
cat >"$launcher" <<'EOF'
#!/bin/sh
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
export DS_HCGE=1 DS_HCGE_DIAG=0 DS_LOG_FLUSH=1
export DS_MIPS_JIT=1 DS_MIPS_NATIVE=1 DS_MIPS_NATIVE_OP=4 DS_MIPS_NATIVE_LIMIT=$limit
export HOME="$here/data"
mkdir -p "$HOME"
exec "$here/lib/ld.so.1" --library-path "$here/lib:/mnt/sdcard/cubegm/usr/lib:/mnt/sdcard/cubegm/lib" "$here/dsperate" "$@" --no-mic --no-audio
EOF
chmod 0755 "$launcher"
cp "$src" "$dst/dsperate"
cp "$launcher" "$dst/run_sf3000.sh"
sync
src_hash=$(sha256sum "$src" | awk '{print $1}')
dst_hash=$(sha256sum "$dst/dsperate" | awk '{print $1}')
[ "$src_hash" = "$dst_hash" ] || { echo "binary verification failed" >&2; exit 1; }
echo "DEPLOYED: $dst/dsperate"
echo "sha256: $dst_hash"
