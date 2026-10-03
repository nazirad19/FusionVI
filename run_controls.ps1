# Control arms for the missing-protein benchmark (Experiment 4).
# Usage:  .\run_controls.ps1            # tier 1 arms + baselines, all control_seeds
#         .\run_controls.ps1 -Tier 2    # also tier 2 arms
# Completed runs are skipped, so the existing totalvi/fusionvi seeds 2026-2029 are reused.
param(
    [int]$Tier = 1,
    [int[]]$Seeds,
    [string]$PythonPath
)
$ErrorActionPreference = 'Stop'
$env:MPLCONFIGDIR = Join-Path $PSScriptRoot '.mpl'
$localPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
$workspacePython = Join-Path $PSScriptRoot '..\work\vaxvi\.venv\Scripts\python.exe'
$python = if ($PythonPath) { $PythonPath } elseif (Test-Path $localPython) { $localPython } else { $workspacePython }
if (-not (Test-Path $python)) { throw 'Create .venv and install requirements.txt before running.' }
$src = Join-Path $PSScriptRoot 'src'

& $python (Join-Path $src 'download_data.py')
$data = Join-Path $PSScriptRoot 'data\processed\paper_figure3_missing_protein.h5ad'
if (-not (Test-Path $data)) { & $python (Join-Path $src 'prepare_paper_benchmark.py') }

$cfg = & $python -c "import yaml,json;c=yaml.safe_load(open(r'$PSScriptRoot\config\paper_benchmark.yaml'));print(json.dumps({'arms':{k:v.get('tier',0) for k,v in c['arms'].items()},'seeds':c['control_seeds']}))" | ConvertFrom-Json
$arms = $cfg.arms.PSObject.Properties | Where-Object { $_.Value -le $Tier } | ForEach-Object { $_.Name }
$runSeeds = if ($Seeds) { $Seeds } else { @($cfg.seeds) }
Write-Host "Arms: $($arms -join ', ')  Seeds: $($runSeeds -join ', ')"

# Seed-major order: every arm gets seed k before any arm gets seed k+1,
# so a partial run still yields balanced paired comparisons.
foreach ($seed in $runSeeds) {
    foreach ($arm in $arms) {
        & $python (Join-Path $src 'train_paper_benchmark.py') --arm $arm --seed $seed
        if ($LASTEXITCODE -ne 0) { throw "Failed: $arm / seed $seed" }
    }
}
Push-Location $src
& $python 'evaluate_controls.py'
Pop-Location
