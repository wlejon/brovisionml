# Download model weights for brovisionml end-to-end testing.
#
# brovisionml loads HuggingFace `safetensors` checkpoints directly (no offline
# conversion step — see CLAUDE.md). This script fetches the right files from the
# HF `resolve` endpoint with Invoke-WebRequest and lays them out under
# `weights\<model>\`. No Python, no HF CLI dependency.
#
# -Model sam-vit-base   (default): facebook/sam-vit-base   (~375 MB) — smallest;
#                                  the quick end-to-end smoke target.
# -Model sam-vit-large            facebook/sam-vit-large  (~1.25 GB)
# -Model sam-vit-huge             facebook/sam-vit-huge   (~2.56 GB) — the
#                                  canonical "SAM", matches SamConfig::vit_h().
#
# All three are promptable-segmentation checkpoints in HF `SamModel` format
# (vision_encoder.* / prompt_encoder.* / mask_decoder.* tensor namespaces, as
# the loaders in src\sam_*.cpp expect — load with the matching SamConfig preset:
# vit_b / vit_l / vit_h). Each repo ships a single un-sharded model.safetensors;
# config.json and preprocessor_config.json come along for reference.
#
# Auth: these repos are public and need no token. For rate-limited or gated
# repos, set $env:HF_TOKEN = "hf_..." and it is sent as a bearer token.
#
# Output: <repo>\weights\<model>\
#   config.json
#   preprocessor_config.json
#   model.safetensors

[CmdletBinding()]
param(
    [ValidateSet("sam-vit-base", "sam-vit-large", "sam-vit-huge",
                 "depth-anything-v2-small", "depth-anything-v2-base",
                 "depth-anything-v2-large")]
    [string]$Model  = "sam-vit-base",
    [string]$Repo   = "",
    [string]$OutDir = "",
    [switch]$Force
)

$ErrorActionPreference = "Stop"

# $SubDir is the on-disk weights\<SubDir> name (HF-cased for the depth models so
# it matches what the loaders/tests look for).
$SubDir = $Model
switch ($Model) {
    "sam-vit-base"  { if (-not $Repo) { $Repo = "facebook/sam-vit-base"  } }
    "sam-vit-large" { if (-not $Repo) { $Repo = "facebook/sam-vit-large" } }
    "sam-vit-huge"  { if (-not $Repo) { $Repo = "facebook/sam-vit-huge"  } }
    "depth-anything-v2-small" {
        if (-not $Repo) { $Repo = "depth-anything/Depth-Anything-V2-Small-hf" }
        $SubDir = "Depth-Anything-V2-Small" }
    "depth-anything-v2-base" {
        if (-not $Repo) { $Repo = "depth-anything/Depth-Anything-V2-Base-hf" }
        $SubDir = "Depth-Anything-V2-Base" }
    "depth-anything-v2-large" {
        if (-not $Repo) { $Repo = "depth-anything/Depth-Anything-V2-Large-hf" }
        $SubDir = "Depth-Anything-V2-Large" }
}
if (-not $OutDir) { $OutDir = Join-Path $PSScriptRoot "..\weights\$SubDir" }

$files = @(
    "config.json",
    "preprocessor_config.json",
    "model.safetensors"
)

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$OutDir = (Resolve-Path $OutDir).Path

Write-Host "Model:   $Model"
Write-Host "Repo:    $Repo"
Write-Host "Target:  $OutDir"
if ($env:HF_TOKEN) { Write-Host "Auth:    HF_TOKEN (bearer)" }
Write-Host ""

$headers = @{}
if ($env:HF_TOKEN) { $headers["Authorization"] = "Bearer $env:HF_TOKEN" }

# fetch <relative-path> <dest-file> [repo] — downloads to <dest>.part then moves
# into place. Throws on any non-success. `repo` defaults to $Repo.
function Fetch([string]$rel, [string]$dest, [string]$repo = $Repo) {
    $url = "https://huggingface.co/$repo/resolve/main/$rel"
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $dest) | Out-Null
    $part = "$dest.part"
    try {
        Invoke-WebRequest -Uri $url -Headers $headers -OutFile $part `
            -MaximumRetryCount 3 -RetryIntervalSec 2 -UseBasicParsing
    } catch {
        if (Test-Path $part) { Remove-Item -Force $part }
        throw "download failed for ${url}: $($_.Exception.Message)"
    }
    Move-Item -Force $part $dest
}

# A $files entry may be a bare path (fetched from $Repo) or "repo|path" to pull
# that one file from a different repo — parity with the sibling repos'
# multi-source models, though every SAM file lives in one repo today.
foreach ($entry in $files) {
    $entRepo = $Repo
    $f = $entry
    if ($entry -match "\|") {
        $parts   = $entry -split "\|", 2
        $entRepo = $parts[0]
        $f       = $parts[1]
    }
    $dest = Join-Path $OutDir $f
    if (-not $Force -and (Test-Path $dest) -and (Get-Item $dest).Length -gt 0) {
        Write-Host "==> $f  (cached, skipping)"
        continue
    }
    Write-Host "==> $f  [$entRepo]"
    Fetch $f $dest $entRepo
}

Write-Host ""
Write-Host "Done. Files in $OutDir :"
Get-ChildItem -Recurse -File $OutDir | Sort-Object FullName | ForEach-Object {
    $rel = $_.FullName.Substring($OutDir.Length + 1)
    "{0,12}  {1}" -f $_.Length, $rel | Write-Host
}
