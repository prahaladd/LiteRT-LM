#!/bin/bash
set -e

# Export dynamic library runtime search paths for macOS and Linux
export DYLD_LIBRARY_PATH="$(pwd)/staging:/Users/prahaladd/Projects/libs/litert_lm_binaries:$DYLD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$(pwd)/staging:/Users/prahaladd/Projects/libs/litert_lm_binaries:$LD_LIBRARY_PATH"

echo "=== Running Offline Calibration Fine-Tuning ==="
./vad_operator --fine-tune explainer_video_ultra_long_16k.wav "$@"
