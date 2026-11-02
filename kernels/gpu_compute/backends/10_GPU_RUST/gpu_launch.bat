REM ============================================================================
REM ABRASAX — Unified Xenoliving System
REM PRIMAL_HEX: 4f5349524953424c58434b | phi: 1.618033988749895
REM CREATOR: Sir Charles Spikes (BASEDGOD)
REM ============================================================================

@echo off
REM 7D Crystal System - GPU Launcher
REM Configures NVRTC path for CUDA GPU acceleration

set NVRTC_PATH=C:\Users\BASEDGOD\AppData\Roaming\Python\Python311\site-packages\nvidia\cuda_nvrtc\bin
set PATH=%NVRTC_PATH%;%PATH%

echo ╔═══════════════════════════════════════════════════════════════════╗
echo ║       7D CRYSTAL SYSTEM - GPU ACCELERATION ENABLED                ║
echo ║                                                                   ║
echo ║   NVRTC: nvidia-cuda-nvrtc-cu12 (pip)                            ║
echo ║   Backend: NVIDIA CUDA GTX 1660 Ti                               ║
echo ╚═══════════════════════════════════════════════════════════════════╝

%*
