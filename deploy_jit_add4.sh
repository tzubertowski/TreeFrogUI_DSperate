#!/bin/sh
set -eu

repo=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
limit=${1:-4}
native=${2:-1}
jit=${3:-1}
card=${CARD_MOUNT:-}
if [ -z "$card" ]; then
  label_path=$(blkid -L R36HD 2>/dev/null || true)
  if [ -d "$label_path" ]; then card=$label_path
  else card=$(findmnt -rn -S "$label_path" -o TARGET 2>/dev/null || true)
  fi
fi
[ -n "$card" ] || [ ! -d /home/${USER}/SDCARD ] || card=/home/${USER}/SDCARD
[ -n "$card" ] || card=/run/media/${USER}/R36HD
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
export DS_SCANLINE_SCALE=1
export DS_MIPS_JIT=__JIT__ DS_MIPS_NATIVE=__NATIVE__ DS_MIPS_NATIVE_OP=all DS_MIPS_NATIVE_LIMIT=__LIMIT__
export HOME="$here/data"
mkdir -p "$HOME"
: >"$here/data/run.log"
echo "dsperate: jit=$DS_MIPS_JIT native=$DS_MIPS_NATIVE limit=$DS_MIPS_NATIVE_LIMIT" >>"$here/data/run.log"
args=""
for a in "$@"; do args="$args [$a]"; done
echo "dsperate: args$args" >>"$here/data/run.log"
# Some SF3000 menu revisions hand standalone launchers the battery-save path
# instead of the content path.  A .sav is never a DS cartridge; recover the
# neighbouring .nds so the core still starts the selected game.
if [ "$#" -eq 1 ] && case "$1" in *.sav|*.SAV) true;; *) false;; esac; then
  rom=${1%.*}.nds
  [ -f "$rom" ] && set -- "$rom"
fi
if "$here/lib/ld.so.1" --library-path "$here/lib:/mnt/sdcard/cubegm/usr/lib:/mnt/sdcard/cubegm/lib" "$here/dsperate" "$@" --no-mic --no-audio >>"$here/data/run.log" 2>&1; then rc=0; else rc=$?; fi
echo "dsperate: exit=$rc" >>"$here/data/run.log"
exit "$rc"
EOF
chmod 0755 "$launcher"
sed -i "s/__JIT__/$jit/; s/__NATIVE__/$native/; s/__LIMIT__/$limit/" "$launcher"
cp "$src" "$dst/dsperate"
cp "$launcher" "$dst/run_sf3000.sh"
sync
src_hash=$(sha256sum "$src" | awk '{print $1}')
dst_hash=$(sha256sum "$dst/dsperate" | awk '{print $1}')
[ "$src_hash" = "$dst_hash" ] || { echo "binary verification failed" >&2; exit 1; }
echo "DEPLOYED: $dst/dsperate"
echo "sha256: $dst_hash"
