#!/bin/bash
# PodAgent health check — verifies all prerequisites before running the pipeline.
# Returns 0 if all checks pass, non-zero if any check fails.

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV_DIR="$PROJECT_ROOT/.venv"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

pass_count=0
warn_count=0
fail_count=0

check_pass() { echo -e "${GREEN}✓${NC} $1"; pass_count=$((pass_count + 1)); }
check_warn() { echo -e "${YELLOW}⚠ ${NC}$1"; warn_count=$((warn_count + 1)); }
check_fail() { echo -e "${RED}✗ ${NC}$1"; fail_count=$((fail_count + 1)); }

echo "=== PodAgent Health Check ==="

# 1. ffmpeg (required for audio processing)
if command -v ffmpeg &>/dev/null; then
    check_pass "ffmpeg found: $(ffmpeg -version | head -1)"
else
    check_fail "ffmpeg not found — required for audio chunking and processing"
fi

# 2. ffprobe (usually bundled with ffmpeg but verify)
if command -v ffprobe &>/dev/null; then
    check_pass "ffprobe found"
else
    check_fail "ffprobe not found — required for duration detection"
fi

# 3. Venv exists and is activated
if [ -d "$VENV_DIR" ] && [ -f "$VENV_DIR/bin/python" ]; then
    check_pass "Python venv exists: $VENV_DIR"
else
    check_fail "Python venv not found at $VENV_DIR — run 'python3 -m venv .venv' and install requirements"
fi

# 4. Venv has required packages
if [ -f "$VENV_DIR/bin/python" ]; then
    for pkg in whisper pyannote.audio torch yt_dlp; do
        import_name="${pkg//-/.}"
        if $VENV_DIR/bin/python -c "import $import_name" &>/dev/null; then
            check_pass "  venv has $pkg"
        else
            check_fail "  venv missing $pkg — install via pip in the venv"
        fi
    done
fi

# 5. GPU detection
if [ -f "$VENV_DIR/bin/python" ]; then
    gpu_info=$($VENV_DIR/bin/python -c "import torch; print('CUDA available:', torch.cuda.is_available(), '|', 'Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')" 2>/dev/null || echo "ERROR")
    if echo "$gpu_info" | grep -q "True"; then
        check_pass "GPU detected: $gpu_info"
    else
        check_warn "No GPU detected — pipeline will run on CPU (2-3x slower)"
    fi
fi

# 6. HuggingFace token configured
if [ -f "$PROJECT_ROOT/config.yaml" ]; then
    hf_raw=$($VENV_DIR/bin/python -c "import yaml; print(yaml.safe_load(open('$PROJECT_ROOT/config.yaml'))['settings']['diarization']['hf_token'])" 2>/dev/null || echo "PARSE_ERROR")
    if [[ "$hf_raw" == *"YOUR_HF_TOKEN"* ]]; then
        check_fail "HuggingFace token is a placeholder — set settings.diarization.hf_token in config.yaml"
    elif [ -n "$hf_raw" ] && [ "$hf_raw" != "PARSE_ERROR" ]; then
        check_pass "HuggingFace token configured (hidden)"
    else
        check_fail "Cannot parse HF token from config.yaml"
    fi
else
    check_fail "config.yaml not found at $PROJECT_ROOT/config.yaml"
fi

# 7. Storage directory exists or is creatable
storage_dir="$PROJECT_ROOT/data"
if [ -d "$storage_dir" ]; then
    check_pass "Storage directory exists: $storage_dir"
elif mkdir -p "$storage_dir" &>/dev/null; then
    check_pass "Created storage directory: $storage_dir"
else
    check_fail "Cannot create storage directory: $storage_dir"
fi

# 8. LLM provider configured (optional — only needed for --analyze)
if [ -f "$PROJECT_ROOT/config.yaml" ]; then
    llm_provider=$($VENV_DIR/bin/python -c "import yaml; c=yaml.safe_load(open('$PROJECT_ROOT/config.yaml')); print(c.get('settings',{}).get('llm',{}).get('provider',''))" 2>/dev/null || echo "")
    if [ -n "$llm_provider" ]; then
        check_pass "LLM provider configured: $llm_provider (optional — needed for --analyze)"
    else
        check_warn "No LLM provider configured — will skip analysis unless you add it to config.yaml"
    fi
fi

# 9. deno (for n-challenge bypass in yt-dlp, optional)
if command -v deno &>/dev/null; then
    check_pass "deno found (used by yt-dlp for JS challenge)"
else
    check_warn "deno not found — some YouTube downloads may fail without it"
fi

echo ""
echo "=== Summary ==="
echo -e "${GREEN}Passed:${NC} $pass_count | ${YELLOW}Warnings:${NC} $warn_count | ${RED}Failed:${NC} $fail_count"

if [ "$fail_count" -gt 0 ]; then
    echo ""
    echo "Pipeline will NOT work until all failures are resolved."
    exit 1
fi

if [ "$warn_count" -gt 0 ]; then
    echo ""
    echo "Pipeline can run but some features may be limited."
fi

exit 0
