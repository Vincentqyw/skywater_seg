# Fetch cuDNN runtime DLLs for Windows GPU builds.
# cuDNN is a runtime dependency of ONNX Runtime CUDA provider.
#
# Sets:
#   CUDNN_BIN_DIR — path to cuDNN DLLs

if(NOT WIN32 OR NOT ONNXRUNTIME_CUDA)
  return()
endif()

set(CUDNN_VERSION "9.1.1.17")
set(_cudnn_name "cudnn-windows-x86_64-${CUDNN_VERSION}_cuda12-archive")
set(_cudnn_url "https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/windows-x86_64/${_cudnn_name}.zip")

include(FetchContent)

message(STATUS "Fetching cuDNN ${CUDNN_VERSION}...")

FetchContent_Declare(cudnn
  URL ${_cudnn_url}
)
FetchContent_MakeAvailable(cudnn)

set(CUDNN_BIN_DIR "${cudnn_SOURCE_DIR}/bin" CACHE PATH "cuDNN runtime DLLs")

message(STATUS "cuDNN fetched: ${cudnn_SOURCE_DIR}")
