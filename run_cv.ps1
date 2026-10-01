$ErrorActionPreference = 'Stop'
$env:MPLCONFIGDIR = Join-Path $PSScriptRoot '.mpl'
$localPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
$workspacePython = Join-Path $PSScriptRoot '..\work\vaxvi\.venv\Scripts\python.exe'
$python = if (Test-Path $localPython) { $localPython } else { $workspacePython }
if (-not (Test-Path $python)) { throw 'Create .venv and install requirements.txt before running.' }
$donors = @(
    '202937150091_R01C01',
    '202937150091_R02C01',
    '202937150118_R01C01',
    '202937150118_R02C01',
    '202937150118_R03C01',
    '202937150118_R04C01',
    '202937150118_R05C01',
    '202937150118_R06C01',
    '202937150118_R07C01',
    '202937150118_R08C01'
)

foreach ($model in @('totalvi', 'fusionvi')) {
    foreach ($donor in $donors) {
        & $python (Join-Path $PSScriptRoot 'src\train_fold.py') `
            --model $model `
            --heldout-donor $donor `
            --seed 2026
        if ($LASTEXITCODE -ne 0) {
            throw "Training failed for $model / $donor"
        }
    }
}
