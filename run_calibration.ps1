# Re-score saved complete-panel models on a common scale (no retraining).
# Writes results/calibration_{protein_metrics,summary,contrasts}.csv and calibration_summary.json.
$ErrorActionPreference = 'Stop'
$python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path $python)) { throw 'Create .venv and install requirements.txt before running.' }
Push-Location (Join-Path $PSScriptRoot 'src')
& $python 'evaluate_calibration.py' @args
if ($LASTEXITCODE -ne 0) { Pop-Location; throw 'Calibration evaluation failed' }
Pop-Location
& $python -m unittest tests.test_calibration
