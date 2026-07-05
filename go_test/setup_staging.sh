#!/bin/bash
set -e

# setup_staging.sh: Sets up whisper.cpp and silero-vad-go dependencies in the staging directory

STAGING_DIR="staging"
mkdir -p "$STAGING_DIR"

echo "=== 1. Cloning dependencies ==="

# Clone whisper.cpp at the correct commit if not present
if [ ! -d "$STAGING_DIR/whisper.cpp" ]; then
    echo "Cloning whisper.cpp..."
    git clone https://github.com/ggerganov/whisper.cpp.git "$STAGING_DIR/whisper.cpp"
    cd "$STAGING_DIR/whisper.cpp"
    # Checkout compatible commit
    git checkout 0ae02cdb2c73
    cd ../..
else
    echo "whisper.cpp already cloned."
fi

# Clone silero-vad-go if not present
if [ ! -d "$STAGING_DIR/silero-vad-go" ]; then
    echo "Cloning silero-vad-go..."
    git clone https://github.com/streamer45/silero-vad-go.git "$STAGING_DIR/silero-vad-go"
else
    echo "silero-vad-go already cloned."
fi

echo ""
echo "=== 2. Downloading Model Files ==="

# Download VAD ONNX model if not present
if [ ! -f "$STAGING_DIR/silero_vad.onnx" ]; then
    echo "Downloading silero_vad.onnx..."
    curl -L -o "$STAGING_DIR/silero_vad.onnx" "https://github.com/snakers4/silero-vad/raw/master/files/silero_vad.onnx"
else
    echo "silero_vad.onnx already present."
fi

# Download Whisper GGML model if not present
if [ ! -f "$STAGING_DIR/ggml-tiny.bin" ]; then
    echo "Downloading ggml-tiny.bin..."
    ./"$STAGING_DIR/whisper.cpp/models/download-ggml-model.sh" tiny
    mv ggml-tiny.bin "$STAGING_DIR/"
else
    echo "ggml-tiny.bin already present."
fi

echo ""
echo "=== 3. How to Compile Shared Libraries for whisper.cpp & ONNX ==="
echo "To finish setup, you must build the shared libraries and download ONNX Runtime."
echo ""
echo "On macOS (Tahoe/Arm64):"
echo "  1. Compile whisper.cpp shared library:"
echo "     cmake -B \"$STAGING_DIR/whisper.cpp/build_go_shared\" -DBUILD_SHARED_LIBS=ON"
echo "     cmake --build \"$STAGING_DIR/whisper.cpp/build_go_shared\" --config Release"
echo "     cp \"$STAGING_DIR/whisper.cpp/build_go_shared/bin/\"*.dylib \"$STAGING_DIR/\""
echo "  2. Download ONNX Runtime tgz, extract it, and copy libonnxruntime.dylib to \"$STAGING_DIR/\""
echo ""
echo "On Linux (Ubuntu/Debian x86_64):"
echo "  1. Compile whisper.cpp shared library:"
echo "     cmake -B \"$STAGING_DIR/whisper.cpp/build_go_shared\" -DBUILD_SHARED_LIBS=ON -DWHISPER_NO_AVX=OFF"
echo "     cmake --build \"$STAGING_DIR/whisper.cpp/build_go_shared\" --config Release"
echo "     cp \"$STAGING_DIR/whisper.cpp/build_go_shared/bin/\"*.so* \"$STAGING_DIR/\""
echo "  2. Download ONNX Runtime for Linux (x64) and copy libonnxruntime.so to \"$STAGING_DIR/\""
echo ""
echo "Setup scripting completed!"
