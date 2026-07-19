# Fetch ONNX Runtime pre-built binaries from GitHub releases.
# Inspired by COLMAP's cmake/FindDependencies.cmake.
#
# Sets:
#   ONNXRUNTIME_ROOTDIR — path to the fetched SDK (include/, lib/)
#   ONNXRUNTIME_BIN_DIR  — path to runtime DLLs / .so / .dylib
#
# Options:
#   ONNXRUNTIME_VERSION  — version tag (default 1.27.1)
#   ONNXRUNTIME_CUDA     — prefer GPU/CUDA package when ON (default OFF)

include(FetchContent)

if(NOT ONNXRUNTIME_VERSION)
  set(ONNXRUNTIME_VERSION "1.27.1")
endif()

# -- Platform / flavour selection --------------------------------------
if(APPLE)
  if(CMAKE_OSX_ARCHITECTURES)
    set(_ort_arch ${CMAKE_OSX_ARCHITECTURES})
  else()
    set(_ort_arch ${CMAKE_SYSTEM_PROCESSOR})
  endif()
  if(_ort_arch STREQUAL "x86_64")
    message(WARNING "ONNX Runtime 1.27+ does not provide macOS x86_64 binaries. "
                    "Set -DFETCH_ONNXRUNTIME=OFF and provide "
                    "-DONNXRUNTIME_ROOTDIR=<path> to use a pre-built SDK.")
    return()
  else()
    set(_ort_name "osx-arm64")
    set(_ort_ext  ".tgz")
    set(_ort_hash "SHA256=e42b77a7281cc6e55141bf44fcfbac2c782b823a491bbb6ac33c781dd991f8a6")
  endif()
elseif(UNIX)
  # Linux
  if(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
    set(_ort_name "linux-aarch64")
    set(_ort_ext  ".tgz")
    set(_ort_hash "SHA256=33c67e33d1e25b816878366ea276589a024f71f000e7ff955c4b33224d639edd")
  else()
    if(ONNXRUNTIME_CUDA)
      set(_ort_name "linux-x64-gpu_cuda12")
      set(_ort_ext  ".tgz")
      set(_ort_hash "SHA256=08b568bd69500c36606aff7c3896ee4fa7d3531719f6b00f43e6a34db41dc4bf")
    else()
      set(_ort_name "linux-x64")
      set(_ort_ext  ".tgz")
      set(_ort_hash "SHA256=25b1ef1fea1acd210d63f8f24dc870ad6e077795ce1f54876252c6d3803c15af")
    endif()
  endif()
elseif(WIN32)
  if(ONNXRUNTIME_CUDA)
    set(_ort_name "win-x64-gpu_cuda12")
    set(_ort_ext  ".zip")
    set(_ort_hash "SHA256=78d4de5ab262f79ac5dd59f08ff0d049b1cea605497f375f8df5ba1a52f26111")
  else()
    set(_ort_name "win-x64")
    set(_ort_ext  ".zip")
    set(_ort_hash "")
  endif()
else()
  message(FATAL_ERROR "Unsupported platform for ONNX Runtime fetch.")
endif()

set(_ort_url "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-${_ort_name}-${ONNXRUNTIME_VERSION}${_ort_ext}")
set(_ort_pkg "onnxruntime-${_ort_name}-${ONNXRUNTIME_VERSION}")

# -- Fetch -------------------------------------------------------------
message(STATUS "Fetching ONNX Runtime ${ONNXRUNTIME_VERSION} (${_ort_name})...")

# When a system-wide CMAKE_TOOLCHAIN_FILE (e.g. vcpkg) is set in the
# environment, FetchContent sub-builds inherit it and may fail.  Clear it
# so the ORT pre-built package is consumed without any toolchain.
set(_ort_cmake_args)
if(DEFINED ENV{CMAKE_TOOLCHAIN_FILE} AND NOT "$ENV{CMAKE_TOOLCHAIN_FILE}" STREQUAL "")
  list(APPEND _ort_cmake_args "-DCMAKE_TOOLCHAIN_FILE=")
endif()

if(_ort_hash)
  FetchContent_Declare(onnxruntime
    URL ${_ort_url}
    URL_HASH ${_ort_hash}
    CMAKE_ARGS ${_ort_cmake_args}
  )
else()
  FetchContent_Declare(onnxruntime
    URL ${_ort_url}
    CMAKE_ARGS ${_ort_cmake_args}
  )
endif()

FetchContent_MakeAvailable(onnxruntime)

# -- Export paths ------------------------------------------------------
set(ONNXRUNTIME_ROOTDIR "${onnxruntime_SOURCE_DIR}" CACHE PATH "ONNX Runtime SDK root")
# For ORT 1.27+, DLLs / .so are in lib/ (not bin/)
set(ONNXRUNTIME_BIN_DIR "${ONNXRUNTIME_ROOTDIR}/lib" CACHE PATH "ONNX Runtime runtime binaries")

message(STATUS "ONNX Runtime fetched: ${ONNXRUNTIME_ROOTDIR}")
