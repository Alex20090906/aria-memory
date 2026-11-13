Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -like '*memsvc.py*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
Start-Sleep 2
schtasks /run /tn fmem_memsvc | Out-Null
Start-Sleep 8
try { (Invoke-RestMethod -Uri http://127.0.0.1:8094/health -TimeoutSec 10) | ConvertTo-Json -Compress } catch { "memsvc not up: $_" }
