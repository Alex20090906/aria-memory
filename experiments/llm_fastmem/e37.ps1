# E37 end-to-end (P8): semantic label finder with ask-back, with and without canonical labels.
# Base = F_stage_log (staging + event log), the best Chinese real-speech setup so far.
# Optional arg: script set subfolder (Z_SET) and a tag suffix; runs listed in $runs.
param([string]$set = "", [string]$suffix = "", [string]$runs = "find,canon_find")
$fm = "<thinker-workdir>\fmem"
Set-Location $fm
$py = "$fm\terminal\python\python.exe"
$env:PYTHONUTF8 = "1"; $env:FMEM_ANCHOR_OFFSET = "0"; $env:FMEM_ERR_DRIVEN = "1"; $env:FMEM_KEY = "dual"; $env:FMEM_SHIFT_NOTOK = "1"
$env:Z_TOPIC = "1"; $env:Z_RELISTEN = "1"
$env:FMEM_ROWSCALE_FILE = "$fm\rowscale.bin"; $env:FMEM_DIRTY = "1"
$env:Z_STAGE = "1"; $env:Z_LOG = "1"; $env:Z_STAGE_TTL = "2"; $env:Z_SET = $set; $env:Z_FIND_TAU = "0.6"
foreach ($r in $runs.Split(",")) {
  Remove-Item "$fm\rowscale.bin", "$fm\memory.bin.dirty" -EA SilentlyContinue
  $env:Z_FIND = ""; $env:Z_CANON = "0"
  if ($r -like "*find*") { $env:Z_FIND = "embed" }
  if ($r -like "*canon*") { $env:Z_CANON = "1" }
  $tag = "G_$r$suffix"
  $env:Z_TAG = $tag
  $t = Measure-Command { & $py zh_v3.py *> "zh_$tag.log" }
  "$tag minutes: $([int]$t.TotalMinutes)"
  Select-String -Path "zh_$tag.log" -Pattern "^SUMMARY|Traceback|Error:" | % { $_.Line.Substring(0, [Math]::Min(1200, $_.Line.Length)) }
  "ask-backs: " + (Select-String -Path "zh_$tag.log" -Pattern "ask-back").Count
}
"E37 DONE"
