# E42 end-to-end (P3/P8): retrieve-then-read answering (H28) and rule-based relative dates (H29) on both script sets.
# Base = F_stage_log (staging + event log), the same base as E37. Key fix (H24) is in fmem_anchor.py.
# Runs: "<name>:<set>" with name rr = read + reldate, rd = reldate only.
param([string]$runs = "rr:set2,rr:,rd:set2,rd:")
$fm = "<thinker-workdir>\fmem"
Set-Location $fm
$py = "$fm\terminal\python\python.exe"
$env:PYTHONUTF8 = "1"; $env:FMEM_ANCHOR_OFFSET = "0"; $env:FMEM_ERR_DRIVEN = "1"; $env:FMEM_KEY = "dual"; $env:FMEM_SHIFT_NOTOK = "1"
$env:Z_TOPIC = "1"; $env:Z_RELISTEN = "1"
$env:FMEM_ROWSCALE_FILE = "$fm\rowscale.bin"; $env:FMEM_DIRTY = "1"
$env:Z_STAGE = "1"; $env:Z_LOG = "1"; $env:Z_STAGE_TTL = "2"; $env:Z_FIND = ""; $env:Z_CANON = "0"; $env:Z_READ_K = "5"
foreach ($item in $runs.Split(",")) {
  $name, $set = $item.Split(":")
  Remove-Item "$fm\rowscale.bin", "$fm\memory.bin.dirty" -EA SilentlyContinue
  $env:Z_SET = $set
  $env:Z_RELDATE = "1"
  $env:Z_READ = $(if ($name -eq "rr") { "1" } else { "0" })
  $tag = "H_$name" + $(if ($set) { "_$set" } else { "_old" })
  $env:Z_TAG = $tag
  $t = Measure-Command { & $py zh_v3.py *> "zh_$tag.log" }
  "$tag minutes: $([int]$t.TotalMinutes)"
  Select-String -Path "zh_$tag.log" -Pattern "^SUMMARY|Traceback|Error:" | % { $_.Line.Substring(0, [Math]::Min(1500, $_.Line.Length)) }
}
"E42 DONE"
