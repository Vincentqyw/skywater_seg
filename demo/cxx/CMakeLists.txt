# Sky-Water-Person Segmentation — C++ ONNX Runtime inference
#
# ONNX Runtime (and cuDNN for GPU) are auto-fetched from official servers
# via CMake FetchContent, like COLMAP's FindDependencies.cmake.
#
# Options:
#   -DFETCH_ONNXRUNTIME=OFF   Use pre-installed SDK at ONNXRUNTIME_ROOTDIR
#   -DONNXRUNTIME_CUDA=ON     Fetch GPU/CUDA package + cuDNN (default OFF)

cmake_minimum_required(VERSION 3.18)
project(skywater_infer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(WIN32)
  add_compile_definitions(NOMINMAX _CRT_SECURE_NO_WARNINGS)
  string(APPEND CMAKE_CXX_FLAGS " /W3")
else()
  string(APPEND CMAKE_CXX_FLAGS " -Wall -Wextra")
endif()

# -- Options -----------------------------------------------------------
option(FETCH_ONNXRUNTIME "Download ONNX Runtime automatically" ON)
option(ONNXRUNTIME_CUDA "Use ONNX Runtime GPU/CUDA package" OFF)

# -- Local cmake modules -----------------------------------------------
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# -- Fetch or find ONNX Runtime ----------------------------------------
if(FETCH_ONNXRUNTIME)
  # Try local thirdparty/ SDK first (offline / no-GitHub-access fallback)
  set(_ort_local_candidates "")
  if(ONNXRUNTIME_CUDA)
    list(APPEND _ort_local_candidates
      "${CMAKE_CURRENT_SOURCE_DIR}/../../thirdparty/onnxruntime-win-x64-gpu_cuda12-1.27.1")
  endif()
  list(APPEND _ort_local_candidates
    "${CMAKE_CURRENT_SOURCE_DIR}/../../thirdparty/onnxruntime-win-x64-1.27.1"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../thirdparty/onnxruntime-osx-arm64-1.27.1"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../thirdparty/onnxruntime-linux-x64-1.27.1")
  foreach(_p ${_ort_local_candidates})
    if(EXISTS "${_p}/include/onnxruntime_cxx_api.h")
      set(ONNXRUNTIME_ROOTDIR "${_p}")
      message(STATUS "Using local ONNX Runtime: ${ONNXRUNTIME_ROOTDIR}")
      break()
    endif()
  endforeach()

  # Fall back to FetchContent if no local SDK found
  if(NOT ONNXRUNTIME_ROOTDIR)
    include(FetchOnnxRuntime)
  endif()
else()
  if(NOT ONNXRUNTIME_ROOTDIR)
    if(WIN32)
      set(ONNXRUNTIME_ROOTDIR "C:/Program Files/onnxruntime")
    else()
      set(ONNXRUNTIME_ROOTDIR "/usr/local")
    endif()
  endif()
  message(STATUS "Using pre-installed ONNX Runtime: ${ONNXRUNTIME_ROOTDIR}")
endif()

# -- Fetch cuDNN (Windows GPU only) ------------------------------------
include(FetchCudnn)

# -- Target ------------------------------------------------------------
add_executable(skywater_infer main.cpp)

# Some ORT packages nest headers in include/onnxruntime/ (macOS, Linux)
target_include_directories(skywater_infer PRIVATE
  "${ONNXRUNTIME_ROOTDIR}/include"
  "${ONNXRUNTIME_ROOTDIR}/include/onnxruntime"
  "${CMAKE_CURRENT_SOURCE_DIR}"   # for stb_image.h
)

target_link_directories(skywater_infer PRIVATE "${ONNXRUNTIME_ROOTDIR}/lib")
target_link_libraries(skywater_infer PRIVATE onnxruntime)

# -- Copy runtime binaries to output dir --------------------------------
if(WIN32)
  file(GLOB SDK_DLLS "${ONNXRUNTIME_ROOTDIR}/lib/*.dll")
  if(CUDNN_BIN_DIR)
    file(GLOB CUDNN_DLLS "${CUDNN_BIN_DIR}/*.dll")
    list(APPEND SDK_DLLS ${CUDNN_DLLS})
  endif()
  if(SDK_DLLS)
    add_custom_command(TARGET skywater_infer POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SDK_DLLS} $<TARGET_FILE_DIR:skywater_infer>)
  endif()
elseif(APPLE)
  file(GLOB SDK_LIBS "${ONNXRUNTIME_ROOTDIR}/lib/*.dylib")
  if(SDK_LIBS)
    add_custom_command(TARGET skywater_infer POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SDK_LIBS} $<TARGET_FILE_DIR:skywater_infer>)
  endif()
else()
  file(GLOB SDK_LIBS "${ONNXRUNTIME_ROOTDIR}/lib/*.so*")
  if(SDK_LIBS)
    add_custom_command(TARGET skywater_infer POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SDK_LIBS} $<TARGET_FILE_DIR:skywater_infer>)
  endif()
endif()

# -- Debugger convenience ----------------------------------------------
if(WIN32)
  set_property(TARGET skywater_infer PROPERTY
    VS_DEBUGGER_WORKING_DIRECTORY "$(OutDir)")
endif()
