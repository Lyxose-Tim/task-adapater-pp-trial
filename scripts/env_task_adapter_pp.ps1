# Task-Adapter++ PowerShell environment helper.
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = $ProjectRoot
Set-Location -LiteralPath $ProjectRoot

$Candidates = @(
    'C:\Users\Lyxose\.conda\envs\task_adapter_pp\python.exe',
    'C:\Users\Lyxose\.conda\envs\tsa_mlt\python.exe'
)
$script:TaskAdapterPython = $Candidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if (-not $script:TaskAdapterPython) {
    throw 'Neither conda environment task_adapter_pp nor tsa_mlt was found.'
}
$script:TaskAdapterEnvironment = Split-Path -Parent $script:TaskAdapterPython
$CondaPaths = @(
    $script:TaskAdapterEnvironment,
    (Join-Path $script:TaskAdapterEnvironment 'Library\mingw-w64\bin'),
    (Join-Path $script:TaskAdapterEnvironment 'Library\usr\bin'),
    (Join-Path $script:TaskAdapterEnvironment 'Library\bin'),
    (Join-Path $script:TaskAdapterEnvironment 'Scripts'),
    (Join-Path $script:TaskAdapterEnvironment 'bin')
)
$env:PATH = (($CondaPaths + $env:PATH.Split(';')) | Select-Object -Unique) -join ';'
$env:CONDA_PREFIX = $script:TaskAdapterEnvironment

function Invoke-TaskAdapterPython {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Args)
    & $script:TaskAdapterPython @Args
}

function Invoke-TsaPython {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Args)
    Invoke-TaskAdapterPython @Args
}

Write-Host "PYTHONPATH=$env:PYTHONPATH"
Write-Host "Python=$script:TaskAdapterPython"
Write-Host 'Use: Invoke-TaskAdapterPython scripts/run_fsar.py validate --config configs/innovation3_hmdb51.yaml'
