# VAD Browser Operator: Linux Compilation & Setup Handbook

This guide outlines the complete process to clone, compile, and run the VAD voice browser operator on a new **Linux Laptop**. 

---

## 1. Cloning the Forked Repository

Clone the custom `LiteRT-LM` fork containing the macOS/Linux Bazel build targets and the `go_test/` operator directory:

```bash
git clone https://github.com/prahaladd/LiteRT-LM.git
cd LiteRT-LM
```

---

## 2. Setting Up Third-Party Dependencies

We keep the repository lightweight by excluding large C++ sources and dynamic libraries. Run the setup script inside the `go_test/` directory to clone `whisper.cpp`, `silero-vad-go`, and download the model weight files:

```bash
cd go_test
./setup_staging.sh
```

This script will:
* Create the `go_test/staging/` directory.
* Clone `whisper.cpp` and check out the compatible commit (`0ae02cdb2c73`).
* Clone `silero-vad-go`.
* Download `silero_vad.onnx` and `ggml-tiny.bin` into `go_test/staging/`.

---

## 3. Compiling LiteRT-LM Shared Libraries (via Bazel)

Compile the C-API shared library (`liblitertlm_c.so`) using Bazel.

From the root `LiteRT-LM/` directory, run:
```bash
# Build the GPU/accelerated version:
bazel build //c/litertlm_c_api:litertlm_c

# OR build the CPU-only version:
bazel build //c/litertlm_c_api:litertlm_c_cpu
```

Once built, the compiled shared library `liblitertlm_c.so` (or `liblitertlm_c_cpu.so`) will be output to:
`bazel-bin/c/litertlm_c_api/`

---

## 4. Compiling whisper.cpp Shared Libraries (via CMake)

On Linux, compile the Whisper shared libraries (`libwhisper.so`, `libggml.so`) using CMake:

```bash
# From the go_test/ directory
cmake -B staging/whisper.cpp/build_go_shared -S staging/whisper.cpp -DBUILD_SHARED_LIBS=ON
cmake --build staging/whisper.cpp/build_go_shared --config Release

# Copy the compiled shared libraries to staging/
cp staging/whisper.cpp/build_go_shared/bin/*.so* staging/
```

---

## 5. Downloading ONNX Runtime Shared Library

Since ONNX Runtime is not built from source, download the pre-compiled shared library `.so` for Linux x86_64:

```bash
# From the go_test/ directory
wget https://github.com/microsoft/onnxruntime/releases/download/v1.19.2/onnxruntime-linux-x64-1.19.2.tgz
tar -xvf onnxruntime-linux-x64-1.19.2.tgz

# Copy the shared library to staging/
cp onnxruntime-linux-x64-1.19.2/lib/libonnxruntime.so* staging/

# Clean up
rm -rf onnxruntime-linux-x64-1.19.2 onnxruntime-linux-x64-1.19.2.tgz
```

---

## 6. Compiling the Go Operator

Once all `.so` dynamic libraries are in `go_test/staging/`, compile the Go program:

```bash
# From the go_test/ directory
go build vad_operator.go
```

Go's CGo compiler will automatically compile against the staging source headers and link the staging `.so` libraries, embedding the new local `@rpath` path into the compiled binary.

---

## 7. Running the Operator

When running on Linux, configure the runtime dynamic library link path `LD_LIBRARY_PATH` to point to:
1. `staging/` (contains `libwhisper.so`, `libggml.so`, `libonnxruntime.so`).
2. `../bazel-bin/c/litertlm_c_api/` (contains the compiled LiteRT FFI library `liblitertlm_c.so`).

```bash
# Run the operator
LD_LIBRARY_PATH=staging/:../bazel-bin/c/litertlm_c_api/ ./vad_operator explainer_video_ultra_long_16k.wav
```
