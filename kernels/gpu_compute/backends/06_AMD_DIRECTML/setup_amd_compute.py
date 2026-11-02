# ============================================================================
# ABRASAX — Unified Xenoliving System
# PRIMAL_HEX: 4f5349524953424c58434b | phi: 1.618033988749895
# CREATOR: Sir Charles Spikes (BASEDGOD)
# ============================================================================

import os
import subprocess
import sys

def setup_amd_env():
    print("☢️ SETTING UP AMD (DIRECTML) COMPUTE ENVIRONMENT")
    print("   This enables the AMD iGPU for Machine Learning via DirectML.")
    print("   Creating a dedicated venv to avoid conflict with NVIDIA (CUDA).")
    
    venv_path = "cbm_amd_env"
    
    # 1. Create Vetual Env
    if not os.path.exists(venv_path):
        print(f"-> Creating venv: {venv_path}...")
        subprocess.run([sys.executable, "-m", "venv", venv_path])
    
    # 2. Install torch-directml
    pip_exe = os.path.join(venv_path, "Scripts", "pip.exe")
    print("-> Installing torch-directml...")
    subprocess.run([pip_exe, "install", "torch-directml"])
    
    # 3. Create a test script
    test_script_path = "test_amd_gpu.py"
    with open(test_script_path, "w", encoding="utf-8") as f:
        f.write("""
import torch
import torch_directml
device = torch_directml.device()
print(f"✅ AMD Compute Device Active: {device}")
t = torch.tensor([1, 2, 3]).to(device)
print(f"   Tensor on AMD: {t}")
""")
    
    print("\n✅ SETUP COMPLETE.")
    print(f"   To run AMD tasks:  {os.path.join(venv_path, 'Scripts', 'python.exe')} test_amd_gpu.py")

if __name__ == "__main__":
    setup_amd_env()
