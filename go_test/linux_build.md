# VAD Browser Operator: Linux Compilation & Setup Handbook (Ubuntu)

This guide provides the complete, step-by-step setup to install system prerequisites, clone, compile, and execute the voice browser operator on a Linux system running **Ubuntu**.

---

## 1. Install System Prerequisites

First, update your package repository and install the standard build tools, compilers, Go language runtime, CMake, git, and FFmpeg:

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake wget git ffmpeg golang-go
```

### Install Bazel (via Bazelisk)
The LiteRT-LM core requires **Bazel** (v7+) to compile the C-API shared library. The easiest way to manage Bazel on Ubuntu is using Bazelisk:

```bash
sudo wget -O /usr/local/bin/bazel https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-linux-amd64
sudo chmod +x /usr/local/bin/bazel
```
Verify the installation:
```bash
bazel --version
```

---

## 2. Cloning the Repository

Clone the custom `LiteRT-LM` fork containing the C-API Bazel build targets:

```bash
git clone https://github.com/prahaladd/LiteRT-LM.git
cd LiteRT-LM
```

---

## 3. Pull Third-Party Sub-Dependencies

Change to the `go_test/` directory and execute the staging script. This automatically downloads the Whisper model weights (`ggml-tiny.bin`) and Silero VAD model weights (`silero_vad.onnx`), and checks out the compatible source commits:

```bash
cd go_test
chmod +x setup_staging.sh
./setup_staging.sh
```

---

## 4. Compile LiteRT-LM C-API (via Bazel)

Return to the root `LiteRT-LM/` directory and build the core C-API shared library. For your CPU-only laptop, compile the CPU target:

```bash
# From LiteRT-LM/ root directory
bazel build //c/litertlm_c_api:litertlm_c_cpu
```

Once compilation completes, the shared library `liblitertlm_c_cpu.so` will be created inside:
`bazel-bin/c/litertlm_c_api/`

---

## 5. Compile whisper.cpp Shared Libraries (via CMake)

Compile the Whisper and GGML shared libraries with CMake:

```bash
# From the LiteRT-LM/go_test/ directory
cmake -B staging/whisper.cpp/build_go_shared -S staging/whisper.cpp -DBUILD_SHARED_LIBS=ON
cmake --build staging/whisper.cpp/build_go_shared --config Release

# Copy the compiled libraries to the staging/ directory
cp staging/whisper.cpp/build_go_shared/bin/*.so* staging/
```

---

## 6. Download ONNX Runtime Shared Library

Download the pre-compiled ONNX Runtime shared library for Linux x86_64 and copy it to the staging folder:

```bash
# From the LiteRT-LM/go_test/ directory
wget https://github.com/microsoft/onnxruntime/releases/download/v1.19.2/onnxruntime-linux-x64-1.19.2.tgz
tar -xvf onnxruntime-linux-x64-1.19.2.tgz

# Copy the library to staging/
cp onnxruntime-linux-x64-1.19.2/lib/libonnxruntime.so* staging/

# Clean up temporary downloads
rm -rf onnxruntime-linux-x64-1.19.2 onnxruntime-linux-x64-1.19.2.tgz
```

---

## 7. Compile the Go Operator

Now that all dynamic libraries (`.so` files) are placed in `go_test/staging/`, compile the Go binary:

```bash
# From the LiteRT-LM/go_test/ directory
export CGO_CFLAGS="-I$(pwd)/staging/whisper.cpp/include -I$(pwd)/staging/whisper.cpp/ggml/include -I$(pwd)/staging/onnxruntime-osx-arm64-1.19.2/include"
export CGO_LDFLAGS="-L$(pwd)/staging"

go build -ldflags="-extldflags '-Wl,-rpath,$(pwd)/staging -Wl,-rpath,$(pwd)/../bazel-bin/c/litertlm_c_api'" -o vad_operator vad_operator.go
```

---

## 8. Run the Interactive Calibration & Walkthrough

To run or calibrate the voice browser, execute the binary by pointing your runtime linker `LD_LIBRARY_PATH` to the location of the compiled libraries:

```bash
# 1. Run Calibration (saves coordinates to canva_mappings.json)
LD_LIBRARY_PATH=staging/:../bazel-bin/c/litertlm_c_api/ ./vad_operator --calibrate explainer_video_ultra_long_16k.wav

# 2. Run Fine-Tuning (non-interactively refines step offsets instantly)
LD_LIBRARY_PATH=staging/:../bazel-bin/c/litertlm_c_api/ ./vad_operator --fine-tune explainer_video_ultra_long_16k.wav

# 3. Playback & Record Screen (uses calibrated mappings)
LD_LIBRARY_PATH=staging/:../bazel-bin/c/litertlm_c_api/ ./vad_operator explainer_video_ultra_long_16k.wav
```

---

## 9. FFmpeg Screen Recording Tuning for Older Laptops (e.g., Lenovo Z50-70)

On older laptops with low-power dual-core CPUs (like the Core i5-4210U), recording the screen at high resolutions and framerates (e.g., 1080p 60fps) will saturate the CPU, causing lag and dropped frames in the video.

To ensure smooth walkthrough recordings on this hardware:
1. **Reduce Resolution to 720p**: Scale the input using `-vf "scale=1280:-2"`.
2. **Cap Framerate at 30 FPS**: Force the capture device framerate using `-framerate 30`.
3. **Use Ultrafast Preset**: Configure the H.264 encoder with `-preset ultrafast` to minimize CPU encoding cycles.

### Configured Capture Command Example (Linux x11grab):
```bash
# Capture screen :0.0 at 30fps scaled to 1280x720 using minimal CPU preset
ffmpeg -y -f x11grab -framerate 30 -video_size 1920x1080 -i :0.0 \
       -vf "scale=1280:-2" -c:v libx264 -preset ultrafast -pix_fmt yuv420p \
       output_screen.mp4
```

