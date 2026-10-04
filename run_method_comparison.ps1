# Calibrated method comparison on complete-missing-panel benchmarks.
# 1) export inputs for Seurat, 2) run the paper's Seurat v3 transfer in R,
# 3) score every method raw and after the identical source-fitted calibration.
# Native totalVI checkpoints must exist unless -SkipNeural is used.
param(
    [string[]]$Benchmarks = @('paper'),
    [string]$PythonPath,
    [string]$Rscript = 'Rscript',
    [switch]$SkipSeurat,
    [switch]$SkipNeural
)
$ErrorActionPreference = 'Stop'
$localPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
$workspacePython = Join-Path $PSScriptRoot '..\work\vaxvi\.venv\Scripts\python.exe'
$python = if ($PythonPath) { $PythonPath } elseif (Test-Path $localPython) { $localPython } else { $workspacePython }
if (-not (Test-Path $python)) { throw 'Create .venv and install requirements.txt before running.' }
$src = Join-Path $PSScriptRoot 'src'
& $python (Join-Path $src 'download_data.py')
& $python (Join-Path $src 'prepare_benchmarks.py')
foreach ($b in $Benchmarks) {
    if (-not $SkipSeurat) {
        if (-not (Get-Command $Rscript -ErrorAction SilentlyContinue)) {
            throw "Rscript was not found. Install R with Seurat, or rerun with -SkipSeurat."
        }
        & $python (Join-Path $src 'export_for_seurat.py') --benchmark $b
        & $Rscript (Join-Path $PSScriptRoot 'R\seurat_transfer.R') $b
        if ($LASTEXITCODE -ne 0) { throw "Seurat transfer failed for $b" }
    }
    Push-Location $src
    $comparisonArgs = @('compare_methods_calibrated.py', '--benchmark', $b, '--calib-cells', '0')
    if ($SkipNeural) { $comparisonArgs += '--skip-neural' }
    & $python @comparisonArgs
    if ($LASTEXITCODE -ne 0) { Pop-Location; throw "Comparison failed for $b" }
    Pop-Location
}
