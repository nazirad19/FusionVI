$ErrorActionPreference = 'Stop'
$env:MPLCONFIGDIR = Join-Path $PSScriptRoot '.mpl'
$localPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
$workspacePython = Join-Path $PSScriptRoot '..\work\vaxvi\.venv\Scripts\python.exe'
$python = if (Test-Path $localPython) { $localPython } else { $workspacePython }
if (-not (Test-Path $python)) { throw 'Create .venv and install requirements.txt before running.' }

& $python (Join-Path $PSScriptRoot 'src\download_papalexi.py')
& $python (Join-Path $PSScriptRoot 'src\prepare_papalexi.py')
foreach ($model in @('totalvi', 'fusionvi')) {
    foreach ($fold in 0..4) {
        $complete = Join-Path $PSScriptRoot "results\experiment2_papalexi\folds\${model}__fold${fold}__seed2026\complete.json"
        if (Test-Path $complete) {
            Write-Host "Skipping completed fold $model / $fold"
            continue
        }
        & $python (Join-Path $PSScriptRoot 'src\train_papalexi_fold.py') `
            --model $model `
            --fold $fold `
            --seed 2026
        if ($LASTEXITCODE -ne 0) { throw "Training failed for $model / fold $fold" }
    }
}
& $python (Join-Path $PSScriptRoot 'src\evaluate_papalexi.py')
& $python (Join-Path $PSScriptRoot 'src\make_papalexi_figures.py')
