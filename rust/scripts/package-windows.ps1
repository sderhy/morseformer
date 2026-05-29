param(
    [string] $OnnxDir,
    [switch] $SkipModels
)

$ErrorActionPreference = "Stop"

$RustDir = Resolve-Path (Join-Path $PSScriptRoot "..")
$RepoRoot = Resolve-Path (Join-Path $RustDir "..")
$RuntimeDir = Join-Path $RustDir "morseformer-rt"
$UiDir = Join-Path $RustDir "morseformer-ui"
$ResourceDir = Join-Path $UiDir "src-tauri\resources"
$ModelResourceDir = Join-Path $ResourceDir "models\rnnt_phase11b"
$RuntimeExe = Join-Path $RuntimeDir "target\release\morseformer-rt.exe"
$RuntimeReleaseDir = Join-Path $RuntimeDir "target\release"
$RuntimeDepsDir = Join-Path $RuntimeReleaseDir "deps"

if (-not $OnnxDir) {
    $OnnxDir = Join-Path $RepoRoot "build\onnx\rnnt_phase11b"
}

Write-Host "Building morseformer-rt release binary..."
Push-Location $RuntimeDir
cargo build --release
Pop-Location

if (-not (Test-Path $RuntimeExe)) {
    throw "Runtime binary was not produced at $RuntimeExe"
}

New-Item -ItemType Directory -Force -Path $ResourceDir | Out-Null
Copy-Item -LiteralPath $RuntimeExe -Destination (Join-Path $ResourceDir "morseformer-rt.exe") -Force

$runtimeDlls = @()
foreach ($dir in @($RuntimeReleaseDir, $RuntimeDepsDir)) {
    if (Test-Path $dir) {
        $runtimeDlls += Get-ChildItem -Path $dir -Filter "onnxruntime*.dll" -File -ErrorAction SilentlyContinue
    }
}
foreach ($dll in $runtimeDlls | Sort-Object -Property FullName -Unique) {
    Copy-Item -LiteralPath $dll.FullName -Destination (Join-Path $ResourceDir $dll.Name) -Force
}
if ($runtimeDlls.Count -eq 0) {
    Write-Warning "No onnxruntime*.dll found next to the release build. If the installed app cannot start inference, copy the ONNX Runtime DLL into src-tauri\resources."
}

if (-not $SkipModels) {
    $requiredModelFiles = @(
        "manifest.json",
        "rnnt_encoder.onnx",
        "rnnt_predictor_step.onnx",
        "rnnt_joint.onnx"
    )

    foreach ($file in $requiredModelFiles) {
        $path = Join-Path $OnnxDir $file
        if (-not (Test-Path $path)) {
            throw "Missing ONNX asset: $path. Pass -OnnxDir <path> or export the model first."
        }
    }

    if (Test-Path $ModelResourceDir) {
        Remove-Item -LiteralPath $ModelResourceDir -Recurse -Force
    }
    New-Item -ItemType Directory -Force -Path $ModelResourceDir | Out-Null
    Get-ChildItem -LiteralPath $OnnxDir -Force | ForEach-Object {
        Copy-Item -LiteralPath $_.FullName -Destination $ModelResourceDir -Recurse -Force
    }
}

Write-Host "Building Tauri NSIS installer..."
Push-Location $UiDir
if (Test-Path "package-lock.json") {
    npm ci
} else {
    npm install
}
npm run tauri -- build --bundles nsis
Pop-Location

Write-Host "Installer output:"
Get-ChildItem -Path (Join-Path $UiDir "src-tauri\target\release\bundle\nsis") -Filter "*.exe" |
    Select-Object -ExpandProperty FullName
