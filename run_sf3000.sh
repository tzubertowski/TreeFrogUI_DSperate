#!/bin/sh
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
export DS_HCGE=1
export DS_HCGE_DIAG=${DS_HCGE_DIAG:-0}
export DS_LOG_FLUSH=1
export DS_MIPS_JIT=${DS_MIPS_JIT:-1}
export DS_MIPS_NATIVE=${DS_MIPS_NATIVE:-1}
export DS_MIPS_NATIVE_LIMIT=${DS_MIPS_NATIVE_LIMIT:-5}
export HOME="$here/data"
mkdir -p "$HOME"
: >"$HOME/run.log"
set -- "$@" --no-mic --no-audio
if [ "${DS_MIPS_JIT:-0}" = 1 ]; then mode=jit; else mode=interp; set -- "$@" --interp; fi
echo "dsperate: mode=$mode args=$*" >>"$HOME/run.log"
if "$here/lib/ld.so.1" --library-path "$here/lib:/mnt/sdcard/cubegm/usr/lib:/mnt/sdcard/cubegm/lib" "$here/dsperate" "$@" >>"$here/data/run.log" 2>&1; then rc=0; else rc=$?; fi
echo "dsperate: exit=$rc" >>"$here/data/run.log"
exit "$rc"
