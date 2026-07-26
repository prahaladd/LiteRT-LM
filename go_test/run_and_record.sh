#!/bin/bash
set -e

# Configuration
WAV_SRC="explainer_video_ultra_long_16k.wav"
OUT_NAME="${1:-my_custom_recording_name.mp4}"
OUT_PATH="output/$OUT_NAME"

# Export dynamic library runtime search paths for macOS and Linux
export DYLD_LIBRARY_PATH="$(pwd)/staging:/Users/prahaladd/Projects/litelmrt/libs/litert_lm_binaries:$DYLD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$(pwd)/staging:/Users/prahaladd/Projects/litelmrt/libs/litert_lm_binaries:$LD_LIBRARY_PATH"

CHROME_USER_DATA_DIR="/Users/prahaladd/Projects/litelmrt/LiteRT-LM/go_test/staging/chrome_dev_profile"
CHROME_PROFILE_DIR="Jarvis"

echo "=== Running Local VAD Browser Operator (Profile: $CHROME_PROFILE_DIR, Output: $OUT_PATH) ==="
if ./vad_operator --output "$OUT_PATH" \
                  --chrome-user-data-dir "$CHROME_USER_DATA_DIR" \
                  --chrome-profile-dir "$CHROME_PROFILE_DIR" \
                  "$WAV_SRC" "${@:2}"; then
    echo "Walkthrough recording completed successfully: $OUT_PATH"
else
    echo "Walkthrough recording failed."
fi
