#!/bin/sh
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
export DS_HCGE=1
export DS_HCGE_DIAG=${DS_HCGE_DIAG:-0}
export DS_LOG_FLUSH=1
export DS_SCANLINE_SCALE=1
export DS_MIPS_JIT=${DS_MIPS_JIT:-1}
export DS_MIPS_NATIVE=${DS_MIPS_NATIVE:-1}
export DS_MIPS_NATIVE_OP=${DS_MIPS_NATIVE_OP:-all}
# Experimental native SUB lowering stays opt-in; it is not release-safe.
export HOME="$here/data"
mkdir -p "$HOME"
: >"$HOME/run.log"
echo "dsperate: requested args=$*" >>"$HOME/run.log"
echo "dsperate: SDL_VIDEODRIVER=${SDL_VIDEODRIVER-unset}" >>"$HOME/run.log"
# Some SF3000 menu revisions hand standalone launchers the battery-save path
# instead of the content path; recover the adjacent ROM before starting.
if [ "$#" -eq 1 ]; then
  case "$1" in
    *.[sS][aA][vV])
      rom=${1%.*}.nds
      if [ -f "$rom" ]; then set -- "$rom"; else echo "dsperate: no matching ROM for save path $1" >>"$HOME/run.log"; exit 2; fi
      ;;
  esac
fi
echo "dsperate: resolved args=$*" >>"$HOME/run.log"
set -- "$@" --no-mic --no-audio
if [ "${DS_MIPS_JIT:-0}" = 1 ]; then mode=jit; else mode=interp; set -- "$@" --interp; fi
echo "dsperate: mode=$mode native=$DS_MIPS_NATIVE op=$DS_MIPS_NATIVE_OP args=$*" >>"$HOME/run.log"
if "$here/lib/ld.so.1" --library-path "$here/lib:/mnt/sdcard/cubegm/usr/lib:/mnt/sdcard/cubegm/lib" "$here/dsperate" "$@" >>"$here/data/run.log" 2>&1; then rc=0; else rc=$?; fi
echo "dsperate: exit=$rc" >>"$here/data/run.log"
exit "$rc"
