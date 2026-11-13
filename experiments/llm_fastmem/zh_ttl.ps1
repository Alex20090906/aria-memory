# Staging window sweep (Gemini round 10 Q3): how many following utterances a heard fact waits before it is committed.
# Default is 2 (F_stage_log). Event log on in every run.
$fm = "<thinker-workdir>\fmem"
Set-Location $fm
$py = "$fm\terminal\python\python.exe"
$env:PYTHONUTF8 = "1"; $env:FMEM_ANCHOR_OFFSET = "0"; $env:FMEM_ERR_DRIVEN = "1"; $env:FMEM_KEY = "dual"; $env:FMEM_SHIFT_NOTOK = "1"
$env:Z_TOPIC = "1"; $env:Z_RELISTEN = "1"
$env:FMEM_ROWSCALE_FILE = "$fm\rowscale.bin"; $env:FMEM_DIRTY = "1"
$env:Z_STAGE = "1"; $env:Z_LOG = "1"
foreach ($ttl in @("1", "4")) {
  Remove-Item "$fm\rowscale.bin", "$fm\memory.bin.dirty" -EA SilentlyContinue
  $tag = "F_stage_log_ttl$ttl"
  $env:Z_TAG = $tag; $env:Z_STAGE_TTL = $ttl
  $t = Measure-Command { & $py zh_v3.py *> "zh_$tag.log" }
  "$tag minutes: $([int]$t.TotalMinutes)"
  Select-String -Path "zh_$tag.log" -Pattern "^SUMMARY|Traceback|Error:" | % { $_.Line.Substring(0, [Math]::Min(800, $_.Line.Length)) }
  "STAGE ops: " + (Select-String -Path "zh_$tag.log" -Pattern "STAGE ").Count
}
