@echo off
REM Build script for skywater_infer
REM ONNX Runtime is auto-fetched from GitHub by CMake (FetchContent).
REM
REM Usage:
REM   build.bat           -> CPU build (default)
REM   build.bat cpu       -> CPU build
REM   build.bat gpu       -> GPU / CUDA build
REM   build.bat clean     -> remove build dir

setlocal enabledelayedexpansion

if "%1"=="clean" (
    echo Cleaning...
    IF EXIST build rmdir /s /q build
    echo Done.
    exit /b 0
)

set "CUDA=OFF"
set "TAG=[CPU]"
if "%1"=="gpu" (
    set "CUDA=ON"
    set "TAG=[GPU / CUDA 12]"
)

set "BUILD_DIR=%~dp0build"

echo ================================================================
echo  Building skywater_infer %TAG%
echo  ONNX Runtime will be auto-fetched from GitHub
echo ================================================================

IF EXIST "%BUILD_DIR%" rmdir /s /q "%BUILD_DIR%"

cmake -B "%BUILD_DIR%" -G "Visual Studio 17 2022" -A x64 ^
    -DCMAKE_TOOLCHAIN_FILE="" ^
    -DONNXRUNTIME_CUDA=%CUDA%
IF %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%

cmake --build "%BUILD_DIR%" --config Release
IF %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%

echo.
echo ================================================================
echo  Build succeeded! %TAG%
echo  Binary: %BUILD_DIR%\Release\skywater_infer.exe
echo.
if "%CUDA%"=="ON" (
echo  GPU: skywater_infer.exe model.onnx input.jpg output.png cuda --iters 100 --overlay overlay.png
echo  NOTE: cuDNN 9 DLLs must be in PATH or exe directory at runtime.
) else (
echo  CPU: skywater_infer.exe model.onnx input.jpg output.png cpu --iters 50 --overlay overlay.png
)
echo ================================================================
