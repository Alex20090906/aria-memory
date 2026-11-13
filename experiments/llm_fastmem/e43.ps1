# E43 end-to-end: E42's best (retrieve-then-read + rule dates) plus the three offline fixes:
#   H30 Chinese night relisten with source sentences (Z_RELISTEN_ZH), H31b sound-twin second read (Z_READ_PHON),
#   H32 any question line routed (Z_ROUTE_ANY). Both script sets.
param([string]$runs = "all:set2,all:")
$fm = "<thinker-workdir>\fmem"
Set-Location $fm
$py = "$fm\terminal\python\python.exe"
$env:PYTHONUTF8 = "1"; $env:PYTHONIOENCODING = "utf-8"
$env:FMEM_ANCHOR_OFFSET = "0"; $env:FMEM_ERR_DRIVEN = "1"; $env:FMEM_KEY = "dual"; $env:FMEM_SHIFT_NOTOK = "1"
$env:Z_TOPIC = "1"; $env:Z_RELISTEN = "1"
$env:FMEM_ROWSCALE_FILE = "$fm\rowscale.bin"; $env:FMEM_DIRTY = "1"
$env:Z_STAGE = "1"; $env:Z_LOG = "1"; $env:Z_STAGE_TTL = "2"; $env:Z_FIND = ""; $env:Z_CANON = "0"; $env:Z_READ_K = "5"
$env:Z_READ = "1"; $env:Z_RELDATE = "1"
foreach ($item in $runs.Split(",")) {
  $name, $set = $item.Split(":")
  Remove-Item "$fm\rowscale.bin", "$fm\memory.bin.dirty" -EA SilentlyContinue
  $env:Z_SET = $set
  $env:Z_RELISTEN_ZH = "1"; $env:Z_READ_PHON = "1"; $env:Z_ROUTE_ANY = "1"
  $tag = "I_$name" + $(if ($set) { "_$set" } else { "_old" })
  $env:Z_TAG = $tag
  $t = Measure-Command { cmd /c "`"$py`" zh_v3.py > zh_$tag.log 2>&1" }
  "$tag minutes: $([int]$t.TotalMinutes)"
  Select-String -Path "zh_$tag.log" -Encoding utf8 -Pattern "^SUMMARY|Traceback|Error:" | % { $_.Line.Substring(0, [Math]::Min(1500, $_.Line.Length)) }
}
"E43 DONE"
