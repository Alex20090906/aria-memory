@echo off
rem Memory service for the three-model loop (local only, 127.0.0.1:8094).
set "PYTHONUTF8=1"
set "FMEM_ANCHOR_OFFSET=0"
set "FMEM_KEY=cross"
set "FMEM_ERR_DRIVEN=1"
set "FMEM_SHIFT_NOTOK=1"
cd /d <thinker-workdir>\fmem
<thinker-workdir>\fmem\terminal\python\python.exe memsvc.py > <thinker-workdir>\fmem\memsvc.log 2>&1
