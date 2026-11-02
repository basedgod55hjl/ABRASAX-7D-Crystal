# ============================================================================
# ABRASAX — Unified Xenoliving System
# PRIMAL_HEX: 4f5349524953424c58434b | phi: 1.618033988749895
# CREATOR: Sir Charles Spikes (BASEDGOD)
# ============================================================================

import subprocess
import re
import time

def get_gpu_usage():
    try:
        # Run nvidia-smi to get VRAM usage
        # Format: index, memory.used, memory.total
        cmd = ["C:\\Windows\\System32\\nvidia-smi.exe", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader,nounits"]
        output = subprocess.check_output(cmd).decode("utf-8").strip()
        
        gpus = []
        for line in output.split("\n"):
            idx, used, total = map(int, line.split(","))
            percent = (used / total) * 100
            gpus.append({
                "index": idx,
                "used_mb": used,
                "total_mb": total,
                "percent": percent
            })
        return gpus
    except Exception as e:
        print(f"Error querying GPU: {e}")
        return None

def monitor_and_optimize():
    print("GPU Memory ABRASAX Monitor Active...")
    while True:
        stats = get_gpu_usage()
        if stats:
            for gpu in stats:
                print(f"GPU {gpu['index']}: {gpu['used_mb']}/{gpu['total_mb']} MB ({gpu['percent']:.2f}%)")
                
                # If VRAM > 90%, trigger offload or cleanup
                if gpu['percent'] > 90:
                    print(f"!!! CRITICAL VRAM USAGE ({gpu['percent']:.2f}%) !!!")
                    print("Initiating Memory Eviction Protocols...")
                    # Future: Call LM Studio API to unload unused models or clear cache
                    
        time.sleep(30)

if __name__ == "__main__":
    monitor_and_optimize()
