param([switch]$Preview)
$ErrorActionPreference = 'Stop'
$root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..')).TrimEnd('\')
if (-not (Test-Path -LiteralPath (Join-Path $root 'crbot\__init__.py'))) {
    throw 'Project root not found. Keep this script in the project tools folder.'
}
$script:results = @()
$script:freed = [long]0
$script:planned = [long]0
$script:failed = $false

function Get-SafeFiles([string]$path) {
    $full = [IO.Path]::GetFullPath($path)
    if (-not $full.StartsWith($root + '\', [StringComparison]::OrdinalIgnoreCase)) {
        throw "Outside project: $full"
    }
    # Check every ancestor as well as descendants; never traverse junctions.
    $cursor = $full
    while ($cursor.Length -ge $root.Length) {
        $item = Get-Item -LiteralPath $cursor -Force
        if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw "Link skipped: $cursor" }
        if ($cursor -eq $root) { break }
        $cursor = Split-Path $cursor -Parent
    }
    $item = Get-Item -LiteralPath $full -Force
    if (-not $item.PSIsContainer) { return $item }
    $pending = New-Object 'System.Collections.Generic.Stack[string]'
    $pending.Push($full)
    while ($pending.Count) {
        foreach ($child in Get-ChildItem -LiteralPath $pending.Pop() -Force) {
            if ($child.Attributes -band [IO.FileAttributes]::ReparsePoint) {
                throw "Link skipped: $($child.FullName)"
            }
            if ($child.PSIsContainer) { $pending.Push($child.FullName) }
            else { $child }
        }
    }
}

function Clear-Artifact([string]$path) {
    if (-not (Test-Path -LiteralPath $path)) { return }
    $bytes = [long]0
    try {
        $files = @(Get-SafeFiles $path)
        $bytes = [long](($files | Measure-Object Length -Sum).Sum)
        $script:planned += $bytes
        if ($Preview) { $state = 'preview' }
        else {
            # Repeat process inspection immediately before each deletion.
            Assert-Idle
            foreach ($process in @(Get-CimInstance Win32_Process -ErrorAction Stop)) {
                $exe = [string]$process.ExecutablePath
                if ($exe -and ($exe.Equals($path, [StringComparison]::OrdinalIgnoreCase) -or
                    $exe.StartsWith($path.TrimEnd('\') + '\', [StringComparison]::OrdinalIgnoreCase))) {
                    throw "Target in use by PID $($process.ProcessId)"
                }
            }
            Remove-Item -LiteralPath $path -Recurse -Force -ErrorAction Stop
            if (Test-Path -LiteralPath $path) { throw 'Target still exists' }
            $script:freed += $bytes
            $state = 'deleted'
        }
        Write-Host ('{0}: {1} ({2:N3} GiB)' -f $state, $path, ($bytes / 1GB))
        $script:results += [pscustomobject]@{path=$path; bytes=$bytes; status=$state}
    } catch {
        $script:failed = $true
        Write-Warning "$path : $_"
        $script:results += [pscustomobject]@{path=$path; bytes=$bytes; status='skipped_or_failed'; error="$_"}
    }
}

function Assert-Idle {
    $processes = @(Get-CimInstance Win32_Process -ErrorAction Stop)
    foreach ($process in $processes) {
        $command = [string]$process.CommandLine
        if (($process.Name -match 'python|pyinstaller|pip|curl|wget') -and
            ((-not $command) -or ($command -match 'build_standalone|build_bundle|PyInstaller|pip\s+install|pip\s+download|gpu_setup|ensure_gpu'))) {
            throw "Build/install process detected (PID $($process.ProcessId)). Wait for it to finish, then run cleanup again."
        }
        $exe = [string]$process.ExecutablePath
        foreach ($area in @('build', 'dist', 'reports')) {
            if ($exe.StartsWith((Join-Path $root $area) + '\', [StringComparison]::OrdinalIgnoreCase)) {
                throw "Program running from cleanup area (PID $($process.ProcessId)). Close it first."
            }
        }
        if ($process.Name -eq 'RoyalLab.exe' -and -not $exe) {
            throw 'Cannot inspect a running RoyalLab process. Close it first.'
        }
    }
}

# Collect using explicit generated locations, never a project-wide extension sweep.
$targets = @()
foreach ($name in @('standalone-runtime','standalone','standalone-resources','exe')) {
    $targets += Join-Path $root "build\$name"
}
$build = Join-Path $root 'build'
if (Test-Path -LiteralPath $build) {
    Get-SafeFiles $build | Out-Null
    foreach ($dir in Get-ChildItem -LiteralPath $build -Directory) {
        if ($dir.Name -match '-release-\d+\.\d+\.\d+$') {
            $targets += Join-Path $dir.FullName 'build'
        }
    }
}
# These are legacy generated packages, not user data folders.
foreach ($name in @('fast','grid-embedded','release','standalone','standalone-final')) {
    $targets += Join-Path $root "dist\$name"
}
$targets += Join-Path $root 'reports\RoyalLab-before-embedded-grid.exe'
$targets += Join-Path $root 'reports\fast-qa\RoyalLab-3.12.0.exe'
$cache = Join-Path $root 'reports\gpu_setup'
if (Test-Path -LiteralPath $cache) {
    $targets += @(Get-SafeFiles $cache | Where-Object { $_.Extension -in @('.whl','.part','.bin') } | ForEach-Object { $_.FullName })
}

# Keep canonical executable and newest versioned executable. Other root versions
# are disposable only when a canonical executable also exists.
$canonical = Join-Path $root 'RoyalLab.exe'
if (Test-Path -LiteralPath $canonical) {
    $versions = @(Get-ChildItem -LiteralPath $root -File | Where-Object {
        $_.Name -match '^RoyalLab-\d+\.\d+\.\d+\.exe$'
    } | Sort-Object { [version]($_.BaseName -replace '^RoyalLab-', '') } -Descending)
    $targets += @($versions | Select-Object -Skip 1 | ForEach-Object { $_.FullName })
}

try {
    Assert-Idle
    foreach ($target in ($targets | Select-Object -Unique)) { Clear-Artifact $target }
} catch {
    $script:failed = $true
    Write-Warning "$_"
    $script:results += [pscustomobject]@{status='blocked'; error="$_"}
}
$reportDir = Join-Path $root 'reports'
if (Test-Path -LiteralPath $reportDir) {
    # Validate ancestors without traversing the whole reports directory.
    if ((Get-Item -LiteralPath $reportDir -Force).Attributes -band [IO.FileAttributes]::ReparsePoint) {
        throw 'Report directory is a link.'
    }
} else { New-Item -ItemType Directory -Path $reportDir | Out-Null }
$report = Join-Path $reportDir ('artifact-cleanup-' + (Get-Date -Format 'yyyyMMdd-HHmmss-fff') + '.json')
[pscustomobject]@{preview=[bool]$Preview; plannedBytes=$script:planned; freedBytes=$script:freed; incomplete=$script:failed; items=@($script:results)} |
    ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $report -Encoding UTF8
Write-Host ('Planned: {0:N2} GiB; deleted: {1:N2} GiB' -f ($script:planned/1GB), ($script:freed/1GB))
Write-Host "Report: $report"
Write-Host 'Preserved: main/newest EXE, source snapshots, Python/CUDA, models, runs, training, sync, configuration and Git.'
if ($script:failed) { exit 1 }
