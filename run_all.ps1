$ErrorActionPreference = 'Stop'
$env:MPLCONFIGDIR = Join-Path $PSScriptRoot '.mpl'
$localPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
$workspacePython = Join-Path $PSScriptRoot '..\work\vaxvi\.venv\Scripts\python.exe'
$python = if (Test-Path $localPython) { $localPython } else { $workspacePython }
if (-not (Test-Path $python)) { throw 'Create .venv and install requirements.txt before running.' }

& $python (Join-Path $PSScriptRoot 'src\download_data.py')
& $python (Join-Path $PSScriptRoot 'src\prepare_data.py')
foreach ($model in @('totalvi', 'fusionvi')) {
    foreach ($fold in 0..1) {
        $complete = Join-Path $PSScriptRoot "results\final_folds\${model}__mouse${fold}__seed2026\complete.json"
        if (Test-Path $complete) { Write-Host "Skipping completed fold $model / mouse $fold"; continue }
        & $python (Join-Path $PSScriptRoot 'src\train_fold.py') --model $model --fold $fold --seed 2026
        if ($LASTEXITCODE -ne 0) { throw "Training failed for $model / mouse $fold" }
    }
}
& $python (Join-Path $PSScriptRoot 'src\evaluate.py')
& $python (Join-Path $PSScriptRoot 'src\make_figure.py')
