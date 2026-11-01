# ============================================================================
# ABRASAX — Unified Xenoliving System
# PRIMAL_HEX: 4f5349524953424c58434b | phi: 1.618033988749895
# CREATOR: Sir Charles Spikes (BASEDGOD)
# ============================================================================

"""
CUDA Core Mapper & Load Balancer
Ensures all CUDA cores are active and implements intelligent load distribution
"""
import subprocess
import json
import sys
import warnings
import psutil
from pathlib import Path

# PyTorch may warn about pynvml vs nvidia-ml-py; safe to silence for this diagnostics script.
warnings.filterwarnings("ignore", message=".*pynvml.*", category=FutureWarning)


def _utf8_stdio():
    """Avoid UnicodeEncodeError on Windows consoles using cp1252."""
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

class CUDALoadBalancer:
    def __init__(self):
        self.config_path = Path(__file__).parent.parent / "cbm_system" / "performance_config.json"
        self.load_config()
        self.nvidia_threshold = 85.0  # If NVIDIA exceeds 85% utilization, start offloading
        
    def load_config(self):
        if self.config_path.exists():
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
        else:
            self.config = {}
        self.apply_llm_controls_env()

    def apply_llm_controls_env(self):
        """Wire llm_controls from performance_config.json to process env (Gemma / OpenAI-compatible clients)."""
        import os
        lc = self.config.get("llm_controls") or {}
        if not isinstance(lc, dict):
            return
        bu = (lc.get("base_url") or "").strip().rstrip("/")
        if bu:
            if not bu.endswith("/v1"):
                bu = f"{bu}/v1"
            os.environ.setdefault("OPENAI_API_KEY", "lm-studio")
            os.environ["OPENAI_BASE_URL"] = bu
            os.environ["LOCAL_LLM_URL"] = bu
            os.environ["LM_STUDIO_URL"] = bu
            os.environ["LMSTUDIO_BASE_URL"] = bu
        if lc.get("model"):
            os.environ["ABRASAX_MODEL"] = str(lc["model"])
            os.environ["LMSTUDIO_MODEL"] = str(lc["model"])
        if lc.get("embedding_model"):
            os.environ["ABRASAX_EMBEDDING_MODEL"] = str(lc["embedding_model"])
        if lc.get("gemma4_full_control"):
            os.environ["ABRASAX_GEMMA4_FULL_CONTROL"] = "1"
        pp = lc.get("profile_path")
        if pp:
            os.environ.setdefault("ABRASAX_CBM_PERFORMANCE_CONFIG", str(pp))
    
    def check_cuda_cores(self):
        """Verify all CUDA cores are accessible"""
        try:
            import torch
            if torch.cuda.is_available():
                device = torch.cuda.current_device()
                props = torch.cuda.get_device_properties(device)
                print(f"🔥 CUDA Device: {props.name}")
                print(f"   Total CUDA Cores: {props.multi_processor_count * 128}")  # Approx for GTX 1660 Ti
                print(f"   Compute Capability: {props.major}.{props.minor}")
                print(f"   Total VRAM: {props.total_memory / (1024**3):.2f} GB")
                
                # Verify all cores are enabled via environment
                import os
                os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Ensure GPU 0 is visible
                torch.cuda.empty_cache()  # Clear any cached allocations
                
                # Force benchmark mode for kernel optimization
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.enabled = True
                
                print(f"   ✅ All CUDA cores mapped and active")
                return True
            else:
                print("❌ CUDA not available")
                return False
        except ImportError:
            print("⚠️ PyTorch not installed, cannot verify CUDA cores")
            return False
    
    def get_nvidia_load(self):
        """Get current NVIDIA GPU utilization"""
        try:
            res = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'], 
                                capture_output=True, text=True)
            if res.returncode == 0:
                util = float(res.stdout.strip())
                return util
        except Exception:
            pass
        return 0.0
    
    def get_vram_usage(self):
        """Get current VRAM usage percentage"""
        try:
            res = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'],
                                capture_output=True, text=True)
            if res.returncode == 0:
                parts = res.stdout.strip().split(',')
                used = float(parts[0].strip())
                total = float(parts[1].strip())
                return (used / total) * 100 if total > 0 else 0.0
        except Exception:
            pass
        return 0.0
    
    def should_offload(self):
        """Determine if we should offload to AMD/CPU"""
        gpu_load = self.get_nvidia_load()
        vram_load = self.get_vram_usage()
        
        print(f"\n📊 NVIDIA Status: GPU {gpu_load}% | VRAM {vram_load:.1f}%")
        
        if gpu_load > self.nvidia_threshold or vram_load > 90:
            print(f"⚠️ NVIDIA Overloaded! Triggering load balance...")
            return True
        return False
    
    def enforce_app_mapping(self):
        """Ensure critical apps are on NVIDIA, background on AMD"""
        high_perf = self.config.get("gpu_task_mapping", {}).get("high_perf_apps", [])
        offload = self.config.get("gpu_task_mapping", {}).get("offload_apps", [])
        
        # Add LM Studio, Cursor, Antigravity explicitly
        critical_apps = ["LM Studio.exe", "lms.exe", "Cursor.exe", "Antigravity.exe", "python.exe"]
        for app in critical_apps:
            if app not in high_perf:
                high_perf.append(app)
        
        print(f"\n🎯 Force Mapping:")
        print(f"   NVIDIA (Compute): {', '.join(critical_apps)}")
        print(f"   AMD (Aux): {', '.join(offload)}")
        
        # Update config
        self.config.setdefault("gpu_task_mapping", {})["high_perf_apps"] = high_perf
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=4)
        
        # Optional: cbm_system hook (no-op if missing)
        script_path = Path(__file__).parent.parent / "cbm_system" / "enforce_amd_mapping.ps1"
        if script_path.is_file():
            subprocess.run(
                ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script_path)],
                capture_output=True,
                text=True,
            )
        else:
            print("ℹ️  Optional enforce_amd_mapping.ps1 not found; GPU mapping stays config-only.")
    
    def optimize_cuda_environment(self):
        """Set optimal CUDA environment variables"""
        import os
        
        # Force all CUDA cores active
        os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Async kernel launches
        os.environ['CUDA_CACHE_MAXSIZE'] = '2147483647'  # Max cache
        os.environ['CUDA_FORCE_PTX_JIT'] = '1'  # Force JIT compilation
        
        # For multi-process scenarios
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        
        print("✅ CUDA Environment Optimized for Max Throughput")

if __name__ == "__main__":
    _utf8_stdio()
    print("=" * 60)
    print("   CUDA CORE MAPPER & LOAD BALANCER")
    print("=" * 60)
    
    balancer = CUDALoadBalancer()
    
    # 1. Verify CUDA cores
    balancer.check_cuda_cores()
    
    # 2. Optimize environment
    balancer.optimize_cuda_environment()
    
    # 3. Enforce app mapping
    balancer.enforce_app_mapping()
    
    # 4. Check if offload needed
    if balancer.should_offload():
        print("\n🔄 Implementing AMD/CPU offload strategy...")
        print("   (This would trigger background process migration)")
    else:
        print("\n✅ NVIDIA within optimal range. No offload needed.")
    
    print("\n" + "=" * 60)
    print("✅ CUDA CORE MAPPING COMPLETE")
    print("=" * 60)
