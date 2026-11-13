# E40: write key vs recall key with the old and the fixed rule (tokenisation only, no memory write)
$fm = "<thinker-workdir>\fmem"
Set-Location $fm
$py = "$fm\terminal\python\python.exe"
$env:PYTHONUTF8 = "1"
foreach ($k in "cross", "dual") {
  $env:FMEM_KEY = $k
  "== key $k"
  & $py keyfix_test.py 2>&1
  Copy-Item keyfix_result.json "keyfix_result_$k.json"
}
