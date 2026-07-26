#!/bin/bash
set -e

# Export dynamic library runtime search paths for macOS and Linux
export DYLD_LIBRARY_PATH="$(pwd)/staging:/Users/prahaladd/Projects/litelmrt/libs/litert_lm_binaries:$DYLD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$(pwd)/staging:/Users/prahaladd/Projects/litelmrt/libs/litert_lm_binaries:$LD_LIBRARY_PATH"

CHROME_USER_DATA_DIR="/Users/prahaladd/Projects/litelmrt/LiteRT-LM/go_test/staging/chrome_dev_profile"
CHROME_PROFILE_DIR="Jarvis"

echo "=== Starting Interactive Calibration Tool (Profile: $CHROME_PROFILE_DIR) ==="
./vad_operator --calibrate \
               --chrome-user-data-dir "$CHROME_USER_DATA_DIR" \
               --chrome-profile-dir "$CHROME_PROFILE_DIR" \
               explainer_video_ultra_long_16k.wav "$@"
