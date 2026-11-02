REM ============================================================================
REM ABRASAX — Unified Xenoliving System
REM PRIMAL_HEX: 4f5349524953424c58434b | phi: 1.618033988749895
REM CREATOR: Sir Charles Spikes (BASEDGOD)
REM ============================================================================

@echo off
REM ============================================================================
REM  compile_o_to_dll.bat -- Link .o files into a shared library (.dll) for ctypes
REM ============================================================================
REM
REM  Compiles cuda_kernel_wrappers.cu and links it with the existing compiled
REM  .o files (crystal_kernels_v1.o, crystal_kernels_v2.o, physx_crystal_dynamics.o)
REM  into a single DLL that exposes host-callable extern "C" wrapper functions.
REM
REM  The resulting DLL can be loaded by cuda_kernel_loader.py via ctypes.CDLL.
REM
REM  Usage:
REM    compile_o_to_dll.bat              -- full build (wrapper .o + link .dll)
REM    compile_o_to_dll.bat clean       -- remove build artifacts
REM    compile_o_to_dll.bat rebuild     -- clean + build
REM
REM  Prerequisites:
REM    - CUDA Toolkit 12.x+ (nvcc in PATH or CUDA_PATH set)
REM    - Existing .o files in the same directory
REM    - Visual Studio build tools (cl.exe) for Windows linking
REM
REM  Output:
REM    build\crystal_kernels.dll  -- the shared library
REM    build\cuda_kernel_wrappers.o -- compiled wrapper object
REM
REM  © 2026 Sir Charles Spikes | Crystal Architecture
REM ============================================================================

setlocal enabledelayedexpansion

REM ---- Paths ----
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1"
if not exist "%CUDA_PATH%" (
    if defined CUDA_PATH set "CUDA_PATH=%CUDA_PATH%"
)
set "NVCC=%CUDA_PATH%\bin\nvcc.exe"
set "ARCH=sm_75"
set "SRCDIR=%~dp0"
set "BLDDIR=%~dp0build"

if not exist "%NVCC%" (
    echo [FAIL] nvcc not found at %NVCC%
    echo        Set CUDA_PATH to your CUDA Toolkit installation.
    exit /b 1
)

REM ---- Parse target ----
set "TARGET=all"
if /I "%~1"=="clean" set "TARGET=clean"
if /I "%~1"=="rebuild" set "TARGET=rebuild"
if /I "%~1"=="dll" set "TARGET=dll"
if /I "%~1"=="wrapper" set "TARGET=wrapper"

REM ---- Clean ----
if /I "%TARGET%"=="clean" (
    echo Cleaning DLL build artifacts...
    if exist "%BLDDIR%\cuda_kernel_wrappers.o" del "%BLDDIR%\cuda_kernel_wrappers.o"
    if exist "%BLDDIR%\crystal_kernels.dll" del "%BLDDIR%\crystal_kernels.dll"
    if exist "%BLDDIR%\crystal_kernels.lib" del "%BLDDIR%\crystal_kernels.lib"
    if exist "%BLDDIR%\crystal_kernels.exp" del "%BLDDIR%\crystal_kernels.exp"
    echo [OK] Cleaned.
    exit /b 0
)

if /I "%TARGET%"=="rebuild" (
    call :clean
    set "TARGET=all"
)

REM ---- Ensure build dir ----
if not exist "%BLDDIR%" mkdir "%BLDDIR%"

echo.
echo ============================================================================
echo  Crystal CUDA Kernel DLL Builder
echo  Target: %ARCH% (GTX 1660 Ti / Turing)
echo  NVCC:   %NVCC%
echo  Source: %SRCDIR%
echo  Build:  %BLDDIR%
echo ============================================================================
echo.

set "WRAPPER_SRC=%SRCDIR%cuda_kernel_wrappers.cu"
set "WRAPPER_OBJ=%BLDDIR%\cuda_kernel_wrappers.o"
set "V1_OBJ=%SRCDIR%crystal_kernels_v1.o"
set "V2_OBJ=%SRCDIR%crystal_kernels_v2.o"
set "PHYSX_OBJ=%SRCDIR%physx_crystal_dynamics.o"
set "DLL_OUT=%BLDDIR%\crystal_kernels.dll"

REM ---- Check prerequisites ----
echo Checking prerequisites...
set "prereq_ok=1"
if not exist "%WRAPPER_SRC%" (
    echo [FAIL] Wrapper source not found: %WRAPPER_SRC%
    set "prereq_ok=0"
)
if not exist "%V1_OBJ%" (
    echo [FAIL] V1 object not found: %V1_OBJ%
    set "prereq_ok=0"
)
if not exist "%V2_OBJ%" (
    echo [FAIL] V2 object not found: %V2_OBJ%
    set "prereq_ok=0"
)
if not exist "%PHYSX_OBJ%" (
    echo [FAIL] PhysX object not found: %PHYSX_OBJ%
    set "prereq_ok=0"
)
if "!prereq_ok!"=="0" (
    echo.
    echo [FAIL] Missing prerequisite files. Ensure all .cu and .o files exist.
    exit /b 1
)
echo [OK] All prerequisites found.

REM ---- Phase 1: Compile wrapper .cu to .o ----
if /I "%TARGET%"=="wrapper" goto :compile_wrapper

echo.
echo [1/2] Compiling wrapper: cuda_kernel_wrappers.cu --^> .o
"%NVCC%" -O3 -arch=%ARCH% -c "%WRAPPER_SRC%" ^
    -o "%WRAPPER_OBJ%" ^
    -I"%SRCDIR%" ^
    --compiler-options "/MD" 2>&1
if %ERRORLEVEL% neq 0 (
    echo [FAIL] Wrapper compilation failed. Check CUDA/VS integration.
    exit /b %ERRORLEVEL%
)
echo [OK] Wrapper compiled: %WRAPPER_OBJ%

:compile_wrapper
if /I "%TARGET%"=="wrapper" (
    echo.
    echo ============================================================================
    echo  Wrapper object compiled: %WRAPPER_OBJ%
    echo ============================================================================
    exit /b 0
)

REM ---- Phase 2: Link all .o files into .dll ----
echo.
echo [2/2] Linking: .o files --^> crystal_kernels.dll

REM Note: nvcc --shared with .o inputs. nvcc handles host linking via MSVC linker
REM and device code linking via nvlink internally.
"%NVCC%" -O3 --shared -arch=%ARCH% ^
    "%WRAPPER_OBJ%" ^
    "%V2_OBJ%" ^
    "%V1_OBJ%" ^
    "%PHYSX_OBJ%" ^
    -o "%DLL_OUT%" ^
    -I"%SRCDIR%" ^
    --compiler-options "/MD" 2>&1

if %ERRORLEVEL% neq 0 (
    echo [FAIL] DLL linking failed.
    echo.
    echo This may require Visual Studio build tools with the correct
    echo architecture (x64). Try running from a "x64 Native Tools Command
    echo Prompt for VS 2022" or similar.
    echo.
    echo Alternative: use the primary PTX-based loader (cuda_kernel_loader.py)
    echo which does not require a compiled DLL.
    exit /b %ERRORLEVEL%
)

echo [OK] DLL generated: %DLL_OUT%

REM ---- Verify DLL exports ----
echo.
echo Verifying exports...
dumpbin /exports "%DLL_OUT%" 2>nul | findstr /C:"cuda_" >nul
if %ERRORLEVEL% equ 0 (
    echo [OK] DLL contains host-callable wrappers.
    for /f "tokens=*" %%e in ('dumpbin /exports "%DLL_OUT%" 2^>nul ^| findstr /C:"cuda_"') do (
        echo      %%e
    )
) else (
    echo [WARN] Could not verify exports. dumpbin may not be available.
    echo        Try: dumpbin /exports "%DLL_OUT%"
)

if /I "%TARGET%"=="all" (
    echo.
    echo ============================================================================
    echo  BUILD COMPLETE
    echo  DLL: %DLL_OUT%
    echo  cuda_kernel_loader.py will load this DLL via ctypes.CDLL(use_dll=True)
    echo  or use the default PTX-based path automatically.
    echo ============================================================================
)

exit /b 0

:clean
if exist "%BLDDIR%\cuda_kernel_wrappers.o" del "%BLDDIR%\cuda_kernel_wrappers.o"
if exist "%BLDDIR%\crystal_kernels.dll" del "%BLDDIR%\crystal_kernels.dll"
if exist "%BLDDIR%\crystal_kernels.lib" del "%BLDDIR%\crystal_kernels.lib"
if exist "%BLDDIR%\crystal_kernels.exp" del "%BLDDIR%\crystal_kernels.exp"
exit /b 0
