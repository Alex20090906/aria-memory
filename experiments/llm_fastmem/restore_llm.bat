@echo off
rem Restore the normal LLM server server (same steps as the desktop launcher, without the window pause).
set "LOCALAPPDATA=<thinker-workdir>\LLMHome\Local"
set "APPDATA=<thinker-workdir>\LLMHome\Roaming"
set "LLM=%LOCALAPPDATA%\LLM server\terminal\0.1.53"
set "PYTHONUTF8=1"
set "PYTHONNOUSERSITE=1"
set "PYTHONHOME="
set "PYTHONPATH="
set /p LLM_API_KEY=<"%USERPROFILE%\LLM server-API-Key.txt"
"%LLM%\python\python.exe" "%LLM%\llm.py" launch --preset Novel --api-key-env LLM_API_KEY --restart --yes --no-tui > <thinker-workdir>\fmem\restore.log 2>&1
"%LLM%\python\python.exe" "%LLM%\llm.py" launch --status >> <thinker-workdir>\fmem\restore.log 2>&1
