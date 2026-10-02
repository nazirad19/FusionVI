$ErrorActionPreference = 'Stop'
$env:MPLCONFIGDIR = Join-Path $PSScriptRoot '.mpl'
$localPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
$workspacePython = Join-Path $PSScriptRoot '..\work\vaxvi\.venv\Scripts\python.exe'
$python = if (Test-Path $localPython) { $localPython } else { $workspacePython }
if (-not (Test-Path $python)) { throw 'Create .venv and install requirements.txt before running.' }

& $python (Join-Path $PSScriptRoot 'src\download_data.py')
& $python (Join-Path $PSScriptRoot 'src\prepare_data.py')
& (Join-Path $PSScriptRoot 'run_cv.ps1')
& $python (Join-Path $PSScriptRoot 'src\classical_baselines.py')
& $python (Join-Path $PSScriptRoot 'src\evaluate.py')
& $python (Join-Path $PSScriptRoot 'src\cross_modal_head.py')
& $python (Join-Path $PSScriptRoot 'src\make_cross_modal_figures.py')
& (Join-Path $PSScriptRoot 'run_experiment2.ps1')
& (Join-Path $PSScriptRoot 'run_experiment3.ps1')
