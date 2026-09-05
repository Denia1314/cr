$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path -LiteralPath $PSScriptRoot).Path
$runsPath = (Resolve-Path -LiteralPath (Join-Path $projectRoot "runs")).Path
$expectedPrefix = $projectRoot.TrimEnd([IO.Path]::DirectorySeparatorChar) + [IO.Path]::DirectorySeparatorChar

if (-not $runsPath.StartsWith($expectedPrefix, [StringComparison]::OrdinalIgnoreCase)) {
    throw "runs directory is outside the project workspace: $runsPath"
}

$targetDirectories = Get-ChildItem -LiteralPath $runsPath -Directory | ForEach-Object {
    @(
        (Join-Path $_.FullName "samples"),
        (Join-Path $_.FullName "frames")
    )
} | Where-Object {
    Test-Path -LiteralPath $_
} | ForEach-Object {
    (Resolve-Path -LiteralPath $_).Path
}

foreach ($directory in $targetDirectories) {
    $leaf = Split-Path $directory -Leaf
    if (
        -not $directory.StartsWith($runsPath + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase) -or
        $leaf -notin @("samples", "frames")
    ) {
        throw "Refusing unexpected cleanup target: $directory"
    }
}

$files = @($targetDirectories | ForEach-Object {
    Get-ChildItem -LiteralPath $_ -File -Filter "*.jpg"
})
$bytes = ($files | Measure-Object Length -Sum).Sum

foreach ($file in $files) {
    Remove-Item -LiteralPath $file.FullName -Force
}

$remaining = @($targetDirectories | ForEach-Object {
    Get-ChildItem -LiteralPath $_ -File -Filter "*.jpg" -ErrorAction SilentlyContinue
})
$jsonl = @(Get-ChildItem -LiteralPath $runsPath -File -Recurse -Filter "*.jsonl")

Write-Host "Deleted JPG files: $($files.Count)"
Write-Host ("Freed: {0:N2} GB" -f ($bytes / 1GB))
Write-Host "Remaining target JPG files: $($remaining.Count)"
Write-Host "Preserved JSONL files: $($jsonl.Count)"

if ($remaining.Count -ne 0) {
    throw "Some JPG files could not be removed."
}
