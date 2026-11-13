@echo off
rem Research server: LLM server (patched copy in <thinker-workdir>\fmem\terminal) with the fast-weight memory engine, local only.
set "LOCALAPPDATA=<thinker-workdir>\LLMHome\Local"
set "APPDATA=<thinker-workdir>\LLMHome\Roaming"
set "LLM=<thinker-workdir>\fmem\terminal"
set "PYTHONUTF8=1"
set "PYTHONNOUSERSITE=1"
set "PYTHONHOME="
set "PYTHONPATH="
set "LLM_FMEM_FILE=<thinker-workdir>\fmem\memory.bin"
set "LLM_FMEM_CAPTURE=<thinker-workdir>\fmem\capture.bin"
set "LLM_FMEM_HEAD=1"
set "LLM_FMEM_ANCHOR=<thinker-workdir>\fmem\anchor.bin"
set "LLM_FMEM_DEBUG=1"
set "LLM_FMEM_SHIFT_NOTOK=1"
"%LLM%\python\python.exe" "%LLM%\llm.py" serve --profile <thinker-workdir>\fmem\fmem.json --host 127.0.0.1 --port %1 > <thinker-workdir>\fmem\serve%1.log 2>&1
