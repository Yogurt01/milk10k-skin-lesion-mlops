#!/usr/bin/env bash
# Download the model package (final-P1a-kg-onnx) from the public GitHub Release and verify every file against its
# manifest (P1-BSH-02). No token and no gh CLI: plain HTTPS with curl.
#
# Usage: scripts/fetch_model.sh [--check] [DEST_DIR]
#   DEST_DIR  default: MODEL_HOST_DIR as Compose reads it (environment, else .env), else
#             artifacts/model_package/final-P1a-kg-onnx; relative MODEL_HOST_DIR values are relative to the repo root
#   --check   verify DEST_DIR only (no download); exit 1 if any file is missing or differs
#
# Resumable and idempotent: files whose size and sha256 already match are skipped; an interrupted download continues
# from its `.part` file. manifest.json itself is pinned by sha256 below, so a changed Release is detected.
# Environment overrides (for another release or a mirror): MODEL_RELEASE_TAG, MODEL_REPO, MODEL_BASE_URL,
# MODEL_MANIFEST_SHA256 (must match the new manifest).
#
# Licence: the model package is CC BY-NC 4.0, non-commercial use only (derived from ISIC MILK10k, attribution
# "MILK study team"); see NOTICE.md in the package. Research model, not a medical device.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TAG="${MODEL_RELEASE_TAG:-model-v1.0.0}"
REPO="${MODEL_REPO:-Yogurt01/milk10k-skin-lesion-mlops}"
BASE_URL="${MODEL_BASE_URL:-https://github.com/${REPO}/releases/download/${TAG}}"
# sha256 of manifest.json of model-v1.0.0 (the package the parity tests ran on, docs/parity_full.md)
MANIFEST_SHA256="${MODEL_MANIFEST_SHA256:-2857715aa479c162edcbfe24e4f80f3a09f7327fbd28f3df45bc86a008bf939d}"

die() { echo "fetch_model: ERROR: $*" >&2; exit 1; }
log() { echo "fetch_model: $*"; }

CHECK_ONLY=0
if [[ "${1:-}" == "--check" ]]; then CHECK_ONLY=1; shift; fi
[[ "${1:-}" == -* ]] && die "unknown option: $1 (usage: $0 [--check] [DEST_DIR])"

# Without DEST_DIR, use the folder Compose mounts (compose.yaml: ${MODEL_HOST_DIR:-<default>}), resolved the same
# way: the shell variable wins over .env; empty or unset means the default; a relative path is relative to the
# repository root (Compose's project directory). Only a plain or quoted MODEL_HOST_DIR=value line in .env is read.
DEFAULT_DEST="./artifacts/model_package/final-P1a-kg-onnx"
if [[ $# -ge 1 ]]; then
  DEST="$1" DEST_FROM="argument"
else
  if [[ -v MODEL_HOST_DIR ]]; then
    DEST="$MODEL_HOST_DIR" DEST_FROM="MODEL_HOST_DIR (environment)"
  else
    DEST=""
    if [[ -f "$ROOT/.env" ]]; then DEST="$(sed -n 's/^[[:space:]]*MODEL_HOST_DIR=//p' "$ROOT/.env" | tail -n 1)"; fi
    DEST="${DEST%$'\r'}"
    if [[ "$DEST" =~ ^\"(.*)\"$ || "$DEST" =~ ^\'(.*)\'$ ]]; then DEST="${BASH_REMATCH[1]}"; fi
    DEST_FROM="MODEL_HOST_DIR (.env)"
  fi
  if [[ -z "$DEST" ]]; then DEST="$DEFAULT_DEST" DEST_FROM="default"; fi
  [[ "$DEST" == /* ]] || DEST="$ROOT/$DEST"
fi

for tool in curl python3; do command -v "$tool" >/dev/null || die "'$tool' is required"; done
if command -v sha256sum >/dev/null; then
  sha256() { sha256sum "$1" | cut -d' ' -f1; }
else
  command -v shasum >/dev/null || die "'sha256sum' or 'shasum' is required"
  sha256() { shasum -a 256 "$1" | cut -d' ' -f1; }
fi
size_of() { stat -c %s "$1" 2>/dev/null || stat -f %z "$1"; }

download() {  # url, target .part file; resumes a partial .part
  curl --fail --location --silent --show-error --retry 3 --retry-delay 2 --continue-at - --output "$2" "$1" \
    || die "download failed: $1"
}

mkdir -p "$DEST"
DEST="$(cd "$DEST" && pwd)"
log "package directory: $DEST (from $DEST_FROM)"

# 1. manifest.json, pinned by sha256
manifest="$DEST/manifest.json"
if [[ ! -f "$manifest" || "$(sha256 "$manifest")" != "$MANIFEST_SHA256" ]]; then
  ((CHECK_ONLY)) && die "manifest.json missing or its sha256 differs from the pinned value ($TAG)"
  log "downloading manifest.json from $BASE_URL"
  rm -f "$manifest.part"
  download "$BASE_URL/manifest.json" "$manifest.part"
  actual="$(sha256 "$manifest.part")"
  if [[ "$actual" != "$MANIFEST_SHA256" ]]; then
    rm -f "$manifest.part"
    die "manifest.json sha256 $actual != pinned $MANIFEST_SHA256 (Release changed or tampered with?)"
  fi
  mv "$manifest.part" "$manifest"
fi

# 2. every file listed in the manifest: name, size, sha256
entries="$(python3 -c '
import json, sys
for name, info in sorted(json.load(open(sys.argv[1]))["files"].items()):
    print(name, int(info["bytes"]), info["sha256"])
' "$manifest")" || die "cannot read the file list from $manifest"

total=0 present=0 fetched=0 fetched_bytes=0
while read -r name bytes digest; do
  [[ "$name" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || die "unsafe file name in manifest: '$name'"
  total=$((total + 1))
  target="$DEST/$name"
  if [[ -f "$target" ]]; then
    if [[ "$(size_of "$target")" == "$bytes" && "$(sha256 "$target")" == "$digest" ]]; then
      present=$((present + 1))
      continue
    fi
    ((CHECK_ONLY)) && die "$name: size or sha256 differs from the manifest"
    log "$name: size or sha256 differs from the manifest; downloading it again"
    rm -f "$target"
  elif ((CHECK_ONLY)); then
    die "$name: missing"
  fi

  part="$target.part"
  if [[ -f "$part" && "$(size_of "$part")" -ge "$bytes" ]]; then rm -f "$part"; fi  # complete but unverified: restart
  if [[ -f "$part" ]]; then
    log "$name: resuming at $(size_of "$part") of $bytes bytes"
  else
    log "$name: downloading $bytes bytes"
  fi
  download "$BASE_URL/$name" "$part"
  actual_size="$(size_of "$part")"
  actual="$(sha256 "$part")"
  if [[ "$actual_size" != "$bytes" || "$actual" != "$digest" ]]; then
    rm -f "$part"
    die "$name: downloaded $actual_size bytes, sha256 $actual; manifest says $bytes bytes, sha256 $digest"
  fi
  mv "$part" "$target"
  fetched=$((fetched + 1))
  fetched_bytes=$((fetched_bytes + bytes))
done <<<"$entries"

((total > 0)) || die "manifest lists no files"
log "OK: manifest.json + $total files verified against the manifest ($present already present, $fetched downloaded, $fetched_bytes bytes)"
