# Sky-Water-Person Segmentation — C++ vs Python ONNX Inference Demo

C++ ONNX Runtime inference with **zero manual SDK downloads** — ORT and cuDNN
are auto-fetched via CMake `FetchContent` (like COLMAP's strategy).

## Directory Layout

```
demo/
├── cxx/
│   ├── main.cpp              # C++ inference program (CPU / CUDA + overlay)
│   ├── CMakeLists.txt        # CMake build config (FetchContent for ORT + cuDNN)
│   ├── cmake/
│   │   ├── FetchOnnxRuntime.cmake   # Auto-download ORT from GitHub
│   │   └── FetchCudnn.cmake         # Auto-download cuDNN from NVIDIA
│   ├── build.bat             # One-click build Windows
│   ├── build.sh              # One-click build macOS / Linux
│   ├── stb_image.h           # Image I/O (public domain, single-header)
│   └── stb_image_write.h
├── python/
│   └── minimal_onnx_demo.py  # Python ONNX inference + C++ comparison
├── demo.ipynb                # Full walkthrough notebook
└── README.md
```

## Prerequisites

### All platforms
- **CMake 3.18+**
- **C++20 compiler** (MSVC 2022, GCC 12+, Clang 16+, Apple Clang 15+)
- **Git** (for CMake FetchContent)

### Python (for comparison)
via `uv` (already installed):
- `onnxruntime`, `Pillow`, `numpy`

## Build

ONNX Runtime (and cuDNN for GPU) are **automatically downloaded** by CMake.
No manual SDK setup required.

### Windows

```bash
cd demo/cxx

# CPU build (default)
build.bat           # or: build.bat cpu

# GPU / CUDA build
build.bat gpu
```

### macOS / Linux

```bash
cd demo/cxx

# CPU build
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build

# GPU / CUDA build (Linux only)
cmake -B build -DCMAKE_BUILD_TYPE=Release -DONNXRUNTIME_CUDA=ON
cmake --build build
```

## Usage

```bash
cd demo/cxx/build/Release   # or build/ on macOS/Linux

# CPU inference
./skywater_infer model.onnx input.jpg output.png cpu --iters 50 --overlay overlay.png

# GPU inference
./skywater_infer model.onnx input.jpg output.png cuda --iters 100 --overlay overlay.png
```

## How It Works

CMake `FetchContent` downloads pre-built ONNX Runtime binaries from:

| Platform | Package |
|----------|---------|
| Windows x64 CPU | `onnxruntime-win-x64-{ver}.zip` |
| Windows x64 GPU | `onnxruntime-win-x64-gpu_cuda12-{ver}.zip` |
| Linux x64 CPU | `onnxruntime-linux-x64-{ver}.tgz` |
| Linux x64 GPU | `onnxruntime-linux-x64-gpu_cuda12-{ver}.tgz` |
| macOS ARM64 | `onnxruntime-osx-arm64-{ver}.tgz` |

For GPU builds on Windows, cuDNN 9.1 is also auto-fetched from NVIDIA's CDN.

SHA256 hashes (from COLMAP) verify download integrity on all platforms.

## Performance (DSCF8980.jpg, 1920×1280, FP16 ONNX)

| | CPU | GPU (CUDA) | vs Python |
|---|-----|------------|-----------|
| **C++** | ~197 ms | **~14 ms** | 2.9× – 40× |
| **Python** | ~566 ms | — | 1× |

Pixel match C++ vs Python: **98.9%** — differences only at class boundaries
due to bilinear resize implementation.

## Class Colours

| Class      | RGB           |
|------------|---------------|
| Background | (0, 0, 0)     |
| Sky        | (255, 140, 0) |
| Water      | (0, 200, 255) |
| Person     | (255, 60, 60) |
