#!/bin/sh
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
export DS_HCGE=1
export DS_HCGE_DIAG=1
export HOME="$here/data"
mkdir -p "$HOME"
set -- "$@" --no-mic
[ "${DS_MIPS_JIT:-0}" = 1 ] || set -- "$@" --interp
if "$here/lib/ld.so.1" --library-path "$here/lib:/mnt/sdcard/cubegm/usr/lib:/mnt/sdcard/cubegm/lib" "$here/dsperate" "$@" >>"$here/data/run.log" 2>&1; then rc=0; else rc=$?; fi
echo "dsperate: exit=$rc" >>"$here/data/run.log"
exit "$rc"
