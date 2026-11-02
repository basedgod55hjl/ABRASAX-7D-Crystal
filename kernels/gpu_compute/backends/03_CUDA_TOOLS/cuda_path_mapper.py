# ============================================================================
# ABRASAX — Unified Xenoliving System
# PRIMAL_HEX: 4f5349524953424c58434b | phi: 1.618033988749895
# CREATOR: Sir Charles Spikes (BASEDGOD)
# ============================================================================

"""
CUDA Path Finder and Permanent Mapper
Locates CUDA installation and creates permanent environment configuration
"""
import os
import sys
import subprocess
import winreg
from pathlib import Path

class CUDAMapper:
    def __init__(self):
        self.cuda_paths = []
        self.cuda_version = None
        
    def find_cuda_installations(self):
        """Search for CUDA in common locations"""
        print("🔍 Searching for CUDA installations...")
        
        search_paths = [
            Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA"),
            Path("C:/Program Files (x86)/NVIDIA GPU Computing Toolkit/CUDA"),
            Path(os.environ.get("CUDA_PATH", ""))
        ]
        
        for base_path in search_paths:
            if base_path.exists():
                # Find version subdirectories
                for version_dir in base_path.iterdir():
                    if version_dir.is_dir() and version_dir.name.startswith("v"):
                        cuda_path = version_dir
                        bin_path = cuda_path / "bin"
                        lib_path = cuda_path / "lib" / "x64"
                        
                        if bin_path.exists():
                            self.cuda_paths.append({
                                "version": version_dir.name,
                                "path": str(cuda_path),
                                "bin": str(bin_path),
                                "lib": str(lib_path) if lib_path.exists() else None
                            })
                            print(f"  ✅ Found CUDA {version_dir.name}: {cuda_path}")
        
        # Also check if nvidia-smi can detect CUDA
        try:
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
            if result.returncode == 0:
                print(f"  ✅ NVIDIA Driver: Active")
        except Exception:
            print(f"  ⚠️ nvidia-smi not accessible")
        
        return len(self.cuda_paths) > 0
    
    def set_permanent_env_vars(self):
        """Set CUDA environment variables permanently (System-level)"""
        if not self.cuda_paths:
            print("❌ No CUDA installations found")
            return False
        
        # Use the latest version found
        latest_cuda = self.cuda_paths[-1]
        
        print(f"\n🔧 Setting permanent environment variables for CUDA {latest_cuda['version']}...")
        
        env_vars = {
            "CUDA_PATH": latest_cuda["path"],
            "CUDA_PATH_" + latest_cuda["version"].replace("v", "V"): latest_cuda["path"],
            "CUDA_HOME": latest_cuda["path"],
        }
        
        # Set using PowerShell for permanence
        for key, value in env_vars.items():
            try:
                cmd = f'[Environment]::SetEnvironmentVariable("{key}", "{value}", "User")'
                subprocess.run(["powershell", "-Command", cmd], check=True)
                print(f"  ✅ {key} = {value}")
            except Exception:
                print(f"  ⚠️ Failed to set {key}")
        
        # Add CUDA bin to PATH
        try:
            # Get current user PATH
            key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment", 0, winreg.KEY_READ | winreg.KEY_WRITE)
            current_path, _ = winreg.QueryValueEx(key, "Path")
            
            cuda_bin = latest_cuda["bin"]
            if cuda_bin not in current_path:
                new_path = current_path + ";" + cuda_bin
                winreg.SetValueEx(key, "Path", 0, winreg.REG_EXPAND_SZ, new_path)
                print(f"  ✅ Added {cuda_bin} to PATH")
            else:
                print(f"  ℹ️ CUDA bin already in PATH")
            
            winreg.CloseKey(key)
        except Exception as e:
            print(f"  ⚠️ Could not modify PATH: {e}")
        
        return True
    
    def create_cuda_config_file(self):
        """Create a batch file to set CUDA environment"""
        if not self.cuda_paths:
            return
        
        latest_cuda = self.cuda_paths[-1]
        
        config_content = f"""@echo off
REM CUDA Environment Configuration - Auto-generated
REM This file ensures CUDA is always accessible

set CUDA_PATH={latest_cuda['path']}
set CUDA_HOME={latest_cuda['path']}
set PATH=%CUDA_PATH%\\bin;%PATH%

REM CUDA Runtime Optimizations
set CUDA_CACHE_MAXSIZE=2147483647
set CUDA_AUTO_BOOST=1
set CUDA_LAUNCH_BLOCKING=0
set CUDA_DEVICE_ORDER=PCI_BUS_ID
set CUDA_VISIBLE_DEVICES=0

echo ✅ CUDA Environment Loaded
echo    Path: %CUDA_PATH%
echo    Version: {latest_cuda['version']}
"""
        
        output_path = Path("cbm_data/cuda_env.bat")
        output_path.parent.mkdir(exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write(config_content)
        
        print(f"\n📝 CUDA config file created: {output_path}")
        print(f"   Run this before using CUDA applications")
    
    def verify_pytorch_cuda(self):
        """Verify PyTorch can see CUDA"""
        print("\n🧪 Testing PyTorch CUDA detection...")
        
        try:
            import torch
            cuda_available = torch.cuda.is_available()
            
            if cuda_available:
                print(f"  ✅ PyTorch CUDA: AVAILABLE")
                print(f"     Device: {torch.cuda.get_device_name(0)}")
                print(f"     Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
            else:
                print(f"  ❌ PyTorch CUDA: NOT AVAILABLE")
                print(f"     PyTorch may not be built with CUDA support")
                print(f"     Or CUDA drivers need reinstall")
            
            return cuda_available
        except ImportError:
            print(f"  ⚠️ PyTorch not installed")
            return False
    
    def generate_vram_preloader(self):
        """Create script to pre-allocate VRAM on startup"""
        preloader_code = '''"""
VRAM Preloader - Allocates GPU memory to keep it active
Prevents VRAM from being reclaimed by system
"""
import torch
import time

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    
    # Get total VRAM
    total_vram = torch.cuda.get_device_properties(0).total_memory
    target_vram = int(total_vram * 0.9)  # Use 90% to keep GPU active
    
    print(f"🔥 VRAM Preloader Starting...")
    print(f"   Total VRAM: {total_vram/(1024**3):.2f} GB")
    print(f"   Target Allocation: {target_vram/(1024**3):.2f} GB")
    
    try:
        # Allocate large tensor
        preload_size = target_vram // 4  # Float32
        preload_tensor = torch.zeros(preload_size, dtype=torch.float32, device=device)
        
        allocated = torch.cuda.memory_allocated() / (1024**3)
        print(f"   ✅ Allocated: {allocated:.2f} GB")
        print(f"   GPU Memory Reserved - Press Ctrl+C to release")
        
        # Keep alive
        while True:
            time.sleep(60)
            
    except KeyboardInterrupt:
        print("\\n✅ VRAM Preloader stopped")
        del preload_tensor
        torch.cuda.empty_cache()
else:
    print("❌ CUDA not available")
'''
        
        output_path = Path("src/tools/vram_preloader.py")
        with open(output_path, 'w') as f:
            f.write(preloader_code)
        
        print(f"\n💾 VRAM Preloader created: {output_path}")

if __name__ == "__main__":
    print("="*70)
    print("   CUDA PATH FINDER & PERMANENT MAPPER")
    print("="*70 + "\n")
    
    mapper = CUDAMapper()
    
    # 1. Find CUDA
    if mapper.find_cuda_installations():
        # 2. Set permanent environment variables
        mapper.set_permanent_env_vars()
        
        # 3. Create config file
        mapper.create_cuda_config_file()
        
        # 4. Verify PyTorch
        mapper.verify_pytorch_cuda()
        
        # 5. Generate VRAM preloader
        mapper.generate_vram_preloader()
        
        print("\n" + "="*70)
        print("✅ CUDA PERMANENTLY MAPPED")
        print("⚠️ Restart shell/applications to use new environment")
        print("="*70)
    else:
        print("\n❌ No CUDA installation found")
        print("   Install CUDA Toolkit from: https://developer.nvidia.com/cuda-downloads")
