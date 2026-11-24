# Fill XCL's DXIL cache from a readable RPF9 shader package without launching a title.
#
#   .\scripts\precompile-shaders.ps1 -Package 'D:\data\shaders.rpf'
#   .\scripts\precompile-shaders.ps1 -Package 'D:\data\shaders.rpf' -Pipelines
#
# The package is read only. Cached DXIL goes to XCL_DXIL_CACHE when -Cache is supplied,
# otherwise to XCL's default (%LOCALAPPDATA%\xcl\dxil). -Pipelines also asks the GPU
# driver to warm its own compute-pipeline cache; it does not execute shader code.
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Package,
    [string]$Xcl = (Join-Path $PSScriptRoot '..\target\release\xcl.exe'),
    [ValidateSet('all', 'cs', 'vs', 'ps', 'ms')][string]$Only = 'cs',
    [ValidateRange(1, 32)][int]$MemoryWindows = 4,
    [ValidateSet('auto', 'on', 'off')][string]$SplitWave64 = 'auto',
    [ValidateRange(0, [int]::MaxValue)][int]$Limit = 0,
    [ValidateRange(0, 256)][int]$DxcThreads = 0,
    [ValidateRange(0, 256)][int]$PipelineThreads = 0,
    [string]$Cache,
    [switch]$Pipelines,
    [switch]$NoBuild,
    [switch]$PlanOnly
)
$ErrorActionPreference = 'Stop'

$packagePath = (Resolve-Path -LiteralPath $Package).Path
if (-not (Test-Path -LiteralPath $packagePath -PathType Leaf)) {
    throw "shader package is not a file: $packagePath"
}

# Fail early with a useful error instead of asking XCL to scan an unrelated or protected file.
$stream = [System.IO.File]::OpenRead($packagePath)
try {
    $magic = New-Object byte[] 4
    if ($stream.Read($magic, 0, 4) -ne 4 -or [System.Text.Encoding]::ASCII.GetString($magic) -ne '9FPR') {
        throw "shader package is not a readable RPF9 archive: $packagePath"
    }
} finally {
    $stream.Dispose()
}

$repoRoot = Split-Path $PSScriptRoot -Parent
$xclPath = [System.IO.Path]::GetFullPath($Xcl)
$arguments = @('shaders', $packagePath, '--precompile', '--windows', "$MemoryWindows")
if ($Only -ne 'all') { $arguments += @('--only', $Only) }
if ($SplitWave64 -eq 'on') { $arguments += @('--split', '1') }
if ($SplitWave64 -eq 'off') { $arguments += @('--split', '0') }
if ($Limit -gt 0) { $arguments += @('--limit', "$Limit") }
if ($Pipelines) {
    if ($Only -notin @('all', 'cs')) {
        throw '-Pipelines is available only with -Only cs or -Only all'
    }
    $arguments += '--pipelines'
}

$environment = @{}
if ($Cache) { $environment.XCL_DXIL_CACHE = [System.IO.Path]::GetFullPath($Cache) }
if ($DxcThreads -gt 0) { $environment.XCL_DXC_THREADS = "$DxcThreads" }
if ($PipelineThreads -gt 0) { $environment.XCL_PIPELINE_THREADS = "$PipelineThreads" }

Write-Host "Package: $packagePath"
Write-Host "DXIL cache: $(if ($Cache) { $environment.XCL_DXIL_CACHE } else { '%LOCALAPPDATA%\xcl\dxil' })"
Write-Host "Command: `"$xclPath`" $($arguments -join ' ')"
foreach ($entry in $environment.GetEnumerator() | Sort-Object Key) {
    Write-Host "Environment: $($entry.Key)=$($entry.Value)"
}
if ($PlanOnly) { return }

if (-not (Test-Path -LiteralPath $xclPath -PathType Leaf)) {
    if ($NoBuild) { throw "no $xclPath - omit -NoBuild to build it" }
    & cargo build --release -p xcl --manifest-path (Join-Path $repoRoot 'Cargo.toml')
    if ($LASTEXITCODE -ne 0) { throw "cargo build failed with exit code $LASTEXITCODE" }
}

$saved = @{}
try {
    foreach ($entry in $environment.GetEnumerator()) {
        $saved[$entry.Key] = @{
            Present = Test-Path -LiteralPath "Env:$($entry.Key)"
            Value = [System.Environment]::GetEnvironmentVariable($entry.Key, 'Process')
        }
        [System.Environment]::SetEnvironmentVariable($entry.Key, $entry.Value, 'Process')
    }
    & $xclPath @arguments
    if ($LASTEXITCODE -ne 0) { throw "shader precompile failed with exit code $LASTEXITCODE" }
} finally {
    foreach ($entry in $saved.GetEnumerator()) {
        if ($entry.Value.Present) {
            [System.Environment]::SetEnvironmentVariable($entry.Key, $entry.Value.Value, 'Process')
        } else {
            Remove-Item -LiteralPath "Env:$($entry.Key)" -ErrorAction SilentlyContinue
        }
    }
}
