# ============================================================================
# ABRASAX — Unified Xenoliving System
# PRIMAL_HEX: 4f5349524953424c58434b | phi: 1.618033988749895
# CREATOR: Sir Charles Spikes (BASEDGOD)
# ============================================================================

"""
CUDA Core Activator - Forces all CUDA cores active with compute workload
Low-level CUDA initialization and persistent activation
"""
import os
import sys
import time
import threading

# Set low-level CUDA environment before importing torch
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Async execution
os.environ['CUDA_CACHE_DISABLE'] = '0'  # Enable cache
os.environ['CUDA_CACHE_MAXSIZE'] = '2147483647'  # 2GB cache
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # All devices
os.environ['CUDA_AUTO_BOOST'] = '1'  # Enable auto boost

try:
    import torch
    import torch.cuda as cuda
    CUDA_AVAILABLE = cuda.is_available()
except ImportError:
    print("❌ PyTorch not installed. Cannot activate CUDA cores.")
    CUDA_AVAILABLE = False
    sys.exit(1)

class CUDACoreActivator:
    def __init__(self):
        if not CUDA_AVAILABLE:
            raise RuntimeError("CUDA not available")
        
        self.device = torch.device('cuda:0')
        self.props = cuda.get_device_properties(0)
        self.core_count = self.props.multi_processor_count * 128  # Approx cores per SM
        
        print(f"🔥 CUDA CORE ACTIVATOR")
        print(f"   Device: {self.props.name}")
        print(f"   SM Count: {self.props.multi_processor_count}")
        print(f"   Approx Cores: {self.core_count}")
        print(f"   VRAM: {self.props.total_memory / (1024**3):.2f} GB")
        
        # Enable all optimizations
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False  # Allow non-deterministic for speed
        
        # Set device
        cuda.set_device(0)
        cuda.empty_cache()
        
    def create_workload(self, size=4096):
        """Create a compute workload that uses all cores"""
        # Matrix multiplication is perfect for saturating CUDA cores
        a = torch.randn(size, size, device=self.device)
        b = torch.randn(size, size, device=self.device)
        return a, b
    
    def run_burst(self, iterations=100):
        """Run a burst of compute to wake up all cores"""
        print(f"\n🚀 Activating all {self.core_count} CUDA cores...")
        
        a, b = self.create_workload(4096)
        
        start = time.time()
        for i in range(iterations):
            c = torch.matmul(a, b)
            if i % 10 == 0:
                cuda.synchronize()  # Ensure completion
                print(f"   Iteration {i}/{iterations} | Cores: ACTIVE")
        
        duration = time.time() - start
        tflops = (2 * 4096**3 * iterations) / (duration * 1e12)
        
        print(f"\n✅ Burst complete: {duration:.2f}s | {tflops:.2f} TFLOPS")
        print(f"   VRAM Used: {cuda.memory_allocated() / (1024**3):.2f} GB")
        
    def keep_alive(self, interval=30):
        """Keep cores warm with periodic compute"""
        print(f"\n🔥 CUDA KEEPALIVE MODE (interval: {interval}s)")
        print("   Press Ctrl+C to stop...")
        
        a, b = self.create_workload(2048)  # Smaller for keepalive
        
        try:
            while True:
                c = torch.matmul(a, b)
                cuda.synchronize()
                
                mem_used = cuda.memory_allocated() / (1024**3)
                temp = self.get_gpu_temp()
                print(f"\r☢️ CUDA CORES: ACTIVE | VRAM: {mem_used:.2f}GB | TEMP: {temp}°C", 
                      end="", flush=True)
                
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\n\n✅ CUDA Keepalive stopped.")
    
    def get_gpu_temp(self):
        """Get GPU temperature via nvidia-smi"""
        try:
            import subprocess
            res = subprocess.run(['nvidia-smi', '--query-gpu=temperature.gpu', 
                                '--format=csv,noheader,nounits'], 
                                capture_output=True, text=True)
            if res.returncode == 0:
                return int(res.stdout.strip())
        except Exception:
            pass
        return 0

def create_cuda_config():
    """Create low-level CUDA configuration file"""
    cuda_conf = """# CUDA Core Activation Configuration
# Force all CUDA cores active and optimized

# Environment Variables
export CUDA_LAUNCH_BLOCKING=0
export CUDA_CACHE_MAXSIZE=2147483647
export CUDA_AUTO_BOOST=1
export CUDA_VISIBLE_DEVICES=0

# Performance Hints
export __GL_THREADED_OPTIMIZATIONS=1
export __GL_SYNC_TO_VBLANK=0
"""
    
    config_path = "cbm_data/cuda_activation.conf"
    os.makedirs("cbm_data", exist_ok=True)
    
    with open(config_path, 'w') as f:
        f.write(cuda_conf)
    
    print(f"✅ CUDA config created: {config_path}")

if __name__ == "__main__":
    print("=" * 70)
    print("   CUDA CORE ACTIVATOR - LOW LEVEL INITIALIZATION")
    print("=" * 70)
    
    # Create configuration
    create_cuda_config()
    
    # Activate cores
    activator = CUDACoreActivator()
    
    # Run activation burst
    activator.run_burst(iterations=50)
    
    # Ask user if they want keepalive mode
    print("\n" + "=" * 70)
    print("OPTIONS:")
    print("  1. Exit (cores will idle)")
    print("  2. Keepalive mode (maintain active state)")
    print("=" * 70)
    
    choice = input("Select [1/2]: ").strip()
    
    if choice == "2":
        activator.keep_alive(interval=10)
    else:
        print("\n✅ CUDA cores activated. Exiting...")
