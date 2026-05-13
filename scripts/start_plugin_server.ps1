param(
    [string]$Stage1ModelDir = "",
    [string]$Stage2ModelDir = "",
    [string]$BindHost = "127.0.0.1",
    [int]$Port = 8000,
    [ValidateSet("", "cpu", "cuda")]
    [string]$ModelDevice = "",
    [double]$AiThreshold = 0.85,
    [double]$LlmThreshold = 0.92,
    [double]$LlmConfidenceThreshold = 0.97,
    [switch]$NoReload
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()

$repoRoot = Split-Path -Parent $PSScriptRoot
$pythonExe = Join-Path $repoRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $pythonExe)) {
    throw "Virtualenv python was not found at '$pythonExe'."
}

if (-not $Stage1ModelDir) {
    $preferredStage1ModelDir = Join-Path $repoRoot "saved_models\hybrid_current"
    if (Test-Path (Join-Path $preferredStage1ModelDir "artifacts.json")) {
        $Stage1ModelDir = $preferredStage1ModelDir
        $stage1SelectionNote = "Using preferred stage1 hybrid model from saved_models\\hybrid_current."
    }
    else {
        $Stage1ModelDir = Join-Path $repoRoot "saved_models\stage1_kcbert_binary"
        $stage1SelectionNote = "Stage1 hybrid artifacts not found; falling back to saved_models\\stage1_kcbert_binary."
    }
}

if (-not $Stage2ModelDir) {
    $Stage2ModelDir = Join-Path $repoRoot "saved_models\stage2_llm_hybrid_consistency_20260504_text07"
}

$resolvedStage1ModelDir = (Resolve-Path $Stage1ModelDir).Path
$resolvedStage2ModelDir = (Resolve-Path $Stage2ModelDir).Path

$env:STAGE1_MODEL_DIR = $resolvedStage1ModelDir
$env:STAGE2_MODEL_DIR = $resolvedStage2ModelDir
$env:AI_THRESHOLD = $AiThreshold.ToString([System.Globalization.CultureInfo]::InvariantCulture)
$env:LLM_THRESHOLD = $LlmThreshold.ToString([System.Globalization.CultureInfo]::InvariantCulture)
$env:LLM_CONFIDENCE_THRESHOLD = $LlmConfidenceThreshold.ToString([System.Globalization.CultureInfo]::InvariantCulture)

if ($ModelDevice) {
    $env:MODEL_DEVICE = $ModelDevice
}

Set-Location $repoRoot

Write-Host "Starting plugin server with:"
Write-Host "  STAGE1_MODEL_DIR=$env:STAGE1_MODEL_DIR"
if ($stage1SelectionNote) {
    Write-Host "  NOTE=$stage1SelectionNote"
}
Write-Host "  STAGE2_MODEL_DIR=$env:STAGE2_MODEL_DIR"
Write-Host "  AI_THRESHOLD=$env:AI_THRESHOLD"
Write-Host "  LLM_THRESHOLD=$env:LLM_THRESHOLD"
Write-Host "  LLM_CONFIDENCE_THRESHOLD=$env:LLM_CONFIDENCE_THRESHOLD"
if ($env:MODEL_DEVICE) {
    Write-Host "  MODEL_DEVICE=$env:MODEL_DEVICE"
}
Write-Host "  URL=http://$BindHost`:$Port"

$uvicornArgs = @(
    "-m",
    "uvicorn",
    "server.app:app",
    "--host",
    $BindHost,
    "--port",
    $Port.ToString()
)

if (-not $NoReload) {
    $uvicornArgs += "--reload"
}

& $pythonExe @uvicornArgs
