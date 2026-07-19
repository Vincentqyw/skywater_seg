#!/usr/bin/env bash
# Build script for skywater_infer (macOS / Linux)
# ONNX Runtime is auto-fetched from GitHub by CMake (FetchContent).
#
# Usage:
#   ./build.sh           CPU build (default)
#   ./build.sh cpu       CPU build
#   ./build.sh gpu       GPU / CUDA build (Linux only)
#   ./build.sh clean     remove build dir
#   ./build.sh help      show this help

set -euo pipefail

BUILD_DIR="build"
CUDA="OFF"

usage() {
  echo "Usage: $0 [cpu|gpu|clean|help]"
  echo
  echo "  cpu     CPU build (default)"
  echo "  gpu     GPU / CUDA build (Linux x64 only)"
  echo "  clean   Remove build directory"
  echo "  help    Show this help"
  exit 0
}

case "${1:-cpu}" in
  clean)
    echo "Cleaning..."
    rm -rf "$BUILD_DIR"
    echo "Done."
    exit 0
    ;;
  help|--help|-h)
    usage
    ;;
  gpu)
    if [[ "$(uname -s)" == "Darwin" ]]; then
      echo "ERROR: CUDA is not available on macOS."
      echo "  Use CPU build: $0 cpu"
      exit 1
    fi
    CUDA="ON"
    echo "=== Building skywater_infer [GPU / CUDA 12] ==="
    ;;
  cpu|*)
    echo "=== Building skywater_infer [CPU] ==="
    ;;
esac

echo "  ONNX Runtime will be auto-fetched from GitHub"
echo

rm -rf "$BUILD_DIR"

cmake -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DONNXRUNTIME_CUDA="$CUDA"

cmake --build "$BUILD_DIR" --config Release

EXE="$BUILD_DIR/skywater_infer"
if [[ -f "$BUILD_DIR/Release/skywater_infer" ]]; then
  EXE="$BUILD_DIR/Release/skywater_infer"
fi

echo
echo "================================================================"
echo " Build succeeded!"
echo " Binary: $EXE"
echo
if [[ "$CUDA" == "ON" ]]; then
  echo " GPU: $EXE model.onnx input.jpg output.png cuda --iters 100 --overlay overlay.png"
else
  echo " CPU: $EXE model.onnx input.jpg output.png cpu --iters 50 --overlay overlay.png"
fi
echo "================================================================"
