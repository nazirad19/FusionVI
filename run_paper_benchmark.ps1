$ErrorActionPreference = 'Stop'
$env:MPLCONFIGDIR = Join-Path $PSScriptRoot '.mpl'
$localPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
$workspacePython = Join-Path $PSScriptRoot '..\work\vaxvi\.venv\Scripts\python.exe'
$python = if (Test-Path $localPython) { $localPython } else { $workspacePython }
if (-not (Test-Path $python)) { throw 'Create .venv and install requirements.txt before running.' }

& $python (Join-Path $PSScriptRoot 'src\download_data.py')
& $python (Join-Path $PSScriptRoot 'src\prepare_paper_benchmark.py')
$seeds = 2026..2029
foreach ($seed in $seeds) {
    foreach ($model in @('totalvi', 'fusionvi')) {
        & $python (Join-Path $PSScriptRoot 'src\train_paper_benchmark.py') --model $model --seed $seed
        if ($LASTEXITCODE -ne 0) { throw "Paper benchmark failed for $model / seed $seed" }
    }
}
& $python (Join-Path $PSScriptRoot 'src\evaluate_paper_benchmark.py')
if ($LASTEXITCODE -ne 0) { throw 'Paper benchmark evaluation failed' }
& $python (Join-Path $PSScriptRoot 'src\rna_baseline_paper_benchmark.py')
if ($LASTEXITCODE -ne 0) { throw 'RNA ridge baseline failed' }
& $python (Join-Path $PSScriptRoot 'src\evaluate_biological_metrics.py') --generate-missing
if ($LASTEXITCODE -ne 0) { throw 'Biological metric evaluation failed' }
& $python -m unittest discover -s (Join-Path $PSScriptRoot 'tests') -v
if ($LASTEXITCODE -ne 0) { throw 'Benchmark integrity tests failed' }
& $python (Join-Path $PSScriptRoot 'src\write_technical_report.py')
if ($LASTEXITCODE -ne 0) { throw 'Technical report generation failed' }
& $python (Join-Path $PSScriptRoot 'src\write_readme.py')
if ($LASTEXITCODE -ne 0) { throw 'README generation failed' }
& $python (Join-Path $PSScriptRoot 'src\build_report_docx.py')
if ($LASTEXITCODE -ne 0) { throw 'Word report generation failed' }
