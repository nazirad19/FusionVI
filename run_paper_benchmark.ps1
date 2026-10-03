$ErrorActionPreference = 'Stop'
$env:MPLCONFIGDIR = Join-Path $PSScriptRoot '.mpl'
$localPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
$workspacePython = Join-Path $PSScriptRoot '..\work\vaxvi\.venv\Scripts\python.exe'
$python = if (Test-Path $localPython) { $localPython } else { $workspacePython }
if (-not (Test-Path $python)) { throw 'Create .venv and install requirements.txt before running.' }

& $python (Join-Path $PSScriptRoot 'src\download_data.py')
& $python (Join-Path $PSScriptRoot 'src\prepare_paper_benchmark.py')
$seeds = 2026..2030
foreach ($seed in $seeds) {
    foreach ($model in @('totalvi', 'fusionvi')) {
        & $python (Join-Path $PSScriptRoot 'src\train_paper_benchmark.py') --model $model --seed $seed
        if ($LASTEXITCODE -ne 0) { throw "Paper benchmark failed for $model / seed $seed" }
    }
}
& $python (Join-Path $PSScriptRoot 'src\evaluate_paper_benchmark.py')
