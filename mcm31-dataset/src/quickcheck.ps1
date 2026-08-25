# quickcheck.ps1 - 10-second bench sanity check on a freshly saved capture.
# Usage:
#   powershell -File "E:\Ai demod\mcm31-dataset\src\quickcheck.ps1" "<path to CSV>"
#
# This is NOT the full verification (that runs in Python later). It only catches
# the two mistakes that force an immediate recapture: CH2 switched off, and a
# wrong time base.

param([Parameter(Mandatory = $true)][string]$Path)

if (-not (Test-Path -LiteralPath $Path)) {
    Write-Host "FAIL  file not found: $Path" -ForegroundColor Red
    exit 1
}

$sizeMB = [math]::Round((Get-Item -LiteralPath $Path).Length / 1MB, 1)

# Read the header and count rows in one streaming pass (fast on ~30 MB files).
$reader = [System.IO.File]::OpenText($Path)
try {
    $header = $reader.ReadLine()
    $first  = $reader.ReadLine()
    $rows   = 0
    while ($null -ne $reader.ReadLine()) { $rows++ }
}
finally { $reader.Close() }

# rows so far excludes header and the first data line; add the first data line back
if ($null -ne $first) { $rows++ }

$cols = ($header -split ',').Count

Write-Host ""
Write-Host "file   : $(Split-Path -Leaf $Path)  ($sizeMB MB)"
Write-Host "header : $header"
Write-Host "cols   : $cols"
Write-Host "rows   : $('{0:N0}' -f $rows)"
Write-Host ""

$ok = $true

if ($cols -ge 3) {
    Write-Host "PASS  3 columns - CH1 and CH2 were both recording" -ForegroundColor Green
} else {
    Write-Host "FAIL  only $cols column(s) - CH2 WAS OFF. Turn CH2 on and recapture." -ForegroundColor Red
    $ok = $false
}

if ($rows -ge 520000 -and $rows -le 526000) {
    Write-Host "PASS  row count is in range (520,000 - 526,000)" -ForegroundColor Green
} else {
    Write-Host "FAIL  row count $('{0:N0}' -f $rows) is out of range - WRONG TIME BASE." -ForegroundColor Red
    Write-Host "      Set 500 ms/div and Max Sa Rate, then recapture." -ForegroundColor Red
    $ok = $false
}

Write-Host ""
if ($ok) {
    Write-Host ">>> OK - carry on to the next file." -ForegroundColor Green
    exit 0
} else {
    Write-Host ">>> RECAPTURE THIS FILE before continuing." -ForegroundColor Red
    exit 1
}
