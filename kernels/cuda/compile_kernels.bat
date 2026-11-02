REM ============================================================================
REM ABRASAX — Unified Xenoliving System
REM PRIMAL_HEX: 4f5349524953424c58434b | phi: 1.618033988749895
REM CREATOR: Sir Charles Spikes (BASEDGOD)
REM ============================================================================

@echo off
REM ============================================================================
REM  compile_kernels.bat
REM  7D mH-Q Crystal CUDA Kernel Compilation Script
REM  Target: GTX 1660 Ti (sm_75, Turing architecture)
REM
REM  Compiles all translated .cu files to PTX, CUBIN, and a combined DLL.
REM
REM  Usage:
REM    compile_kernels.bat              -- full build (PTX + CUBIN + DLL)
REM    compile_kernels.bat ptx          -- PTX only
REM    compile_kernels.bat cubin        -- CUBIN only
REM    compile_kernels.bat dll          -- shared library only
REM    compile_kernels.bat clean        -- remove build artifacts
REM ============================================================================

setlocal enabledelayedexpansion

set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1"
set "NVCC=%CUDA_PATH%\bin\nvcc.exe"
set "ARCH=sm_75"
set "SRCDIR=%~dp0"
set "BLDDIR=%~dp0build"

if not exist "%BLDDIR%" mkdir "%BLDDIR%"

REM ---- defaults ----
if "%~1"=="" set "TARGET=all"
if not "%~1"=="" set "TARGET=%~1"

if /I "%TARGET%"=="clean" goto :clean

echo.
echo ============================================================================
echo  7D mH-Q Crystal CUDA Kernel Compiler
echo  Target Architecture: %ARCH% (GTX 1660 Ti / Turing)
echo  Source: %SRCDIR%
echo  Build:  %BLDDIR%
echo ============================================================================
echo.

set "V2_SRC=%SRCDIR%crystal_kernels_v2.cu"
set "V1_SRC=%SRCDIR%crystal_kernels_v1.cu"
set "PHYSX_SRC=%SRCDIR%physx_crystal_dynamics.cu"

REM ---- PTX generation ----
if /I "%TARGET%"=="all"     goto :ptx
if /I "%TARGET%"=="ptx"     goto :ptx
if /I "%TARGET%"=="cubin"   goto :cubin
if /I "%TARGET%"=="dll"     goto :dll
goto :unknown

:ptx
echo [1/3] Compiling to PTX...
echo.
echo  -- crystal_kernels_v2.cu --
"%NVCC%" -O3 -arch=%ARCH% -ptx "%V2_SRC%" -o "%BLDDIR%\crystal_kernels_v2.ptx"
if %ERRORLEVEL% neq 0 echo [FAIL] v2 PTX compilation failed! & exit /b %ERRORLEVEL%

echo  -- crystal_kernels_v1.cu --
"%NVCC%" -O3 -arch=%ARCH% -ptx "%V1_SRC%" -o "%BLDDIR%\crystal_kernels_v1.ptx"
if %ERRORLEVEL% neq 0 echo [FAIL] v1 PTX compilation failed! & exit /b %ERRORLEVEL%

echo  -- physx_crystal_dynamics.cu --
"%NVCC%" -O3 -arch=%ARCH% -ptx "%PHYSX_SRC%" -o "%BLDDIR%\physx_crystal_dynamics.ptx"
if %ERRORLEVEL% neq 0 echo [FAIL] physx PTX compilation failed! & exit /b %ERRORLEVEL%

echo [OK] All PTX generated.
if /I "%TARGET%"=="ptx" exit /b 0

:cubin
echo.
echo [2/3] Compiling to CUBIN...
echo.
"%NVCC%" -O3 -arch=%ARCH% -cubin "%V2_SRC%" -o "%BLDDIR%\crystal_kernels_v2.cubin"
if %ERRORLEVEL% neq 0 echo [FAIL] v2 CUBIN compilation failed! & exit /b %ERRORLEVEL%

"%NVCC%" -O3 -arch=%ARCH% -cubin "%V1_SRC%" -o "%BLDDIR%\crystal_kernels_v1.cubin"
if %ERRORLEVEL% neq 0 echo [FAIL] v1 CUBIN compilation failed! & exit /b %ERRORLEVEL%

"%NVCC%" -O3 -arch=%ARCH% -cubin "%PHYSX_SRC%" -o "%BLDDIR%\physx_crystal_dynamics.cubin"
if %ERRORLEVEL% neq 0 echo [FAIL] physx CUBIN compilation failed! & exit /b %ERRORLEVEL%

echo [OK] All CUBIN generated.
if /I "%TARGET%"=="cubin" exit /b 0

:dll
echo.
echo [3/3] Compiling combined shared library...
echo.
REM Merge all sources into one compilation unit for the DLL
"%NVCC%" -O3 -arch=%ARCH% --shared -o "%BLDDIR%\crystal_kernels.dll" ^
    "%V2_SRC%" "%V1_SRC%" "%PHYSX_SRC%" ^
    -I"%SRCDIR%" ^
    -Xcompiler "/MT" ^
    -Xlinker "/EXPORT:manifold_projection_7d_kernel" ^
    -Xlinker "/EXPORT:phi_harmonic_modulation_kernel" ^
    -Xlinker "/EXPORT:cbm_flux_unfold_v2_kernel" ^
    -Xlinker "/EXPORT:crystal_resonance_kernel" ^
    -Xlinker "/EXPORT:quantum_evolution_2d_kernel" ^
    -Xlinker "/EXPORT:flower_of_life_kernel" ^
    -Xlinker "/EXPORT:metatron_cube_kernel" ^
    -Xlinker "/EXPORT:mhq7d_attention" ^
    -Xlinker "/EXPORT:physx_crystal_dynamics_kernel"

if %ERRORLEVEL% neq 0 (
    echo [FAIL] DLL compilation failed!
    echo.
    echo NOTE: Shared library on Windows requires CUDA toolkit with VS integration.
    echo Using PTX + PyCUDA is the recommended deployment path.
    exit /b %ERRORLEVEL%
)
echo [OK] Combined DLL generated: %BLDDIR%\crystal_kernels.dll

if /I "%TARGET%"=="all" (
    echo.
    echo ============================================================================
    echo  BUILD COMPLETE
    echo  PTX:  %BLDDIR%\*.ptx
    echo  CUBIN:%BLDDIR%\*.cubin
    echo  DLL:  %BLDDIR%\crystal_kernels.dll
    echo ============================================================================
)
exit /b 0

:clean
echo Cleaning build artifacts...
if exist "%BLDDIR%" rmdir /s /q "%BLDDIR%"
echo [OK] Cleaned.
exit /b 0

:unknown
echo Unknown target: %TARGET%
echo Valid targets: ptx, cubin, dll, clean
exit /b 1
