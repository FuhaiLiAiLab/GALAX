import torch
import os
import subprocess

def select_best_gpu_device():
    """
    Select the best available GPU device with the most free memory.
    Falls back to CPU if no GPU is available or if there are issues.
    
    Returns:
        str: Device string (e.g., 'cuda:0', 'cuda:1', or 'cpu')
    """
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA is not available. Using CPU.")
        return 'cpu'
    
    # Check if we're in a MIG environment
    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    is_mig_environment = 'MIG-' in cuda_visible_devices
    
    if is_mig_environment:
        print(f"🔧 MIG environment detected: {cuda_visible_devices}")
        print("⚠️  Running in MIG mode with limited memory")
        
        # In MIG mode, PyTorch will only see the assigned MIG slice
        device_count = torch.cuda.device_count()
        if device_count > 0:
            # Test the MIG device
            try:
                device = 'cuda:0'
                torch.cuda.set_device(0)
                props = torch.cuda.get_device_properties(0)
                
                print(f"🔍 MIG Device Info:")
                print(f"  Name: {props.name}")
                print(f"  Total Memory: {props.total_memory / (1024**3):.1f} GB")
                
                # Test if we can allocate memory
                test_tensor = torch.tensor([1.0], device=device)
                allocated = torch.cuda.memory_allocated(0) / (1024**3)
                total = props.total_memory / (1024**3)
                free = total - allocated
                
                print(f"  Available Memory: {free:.1f} GB")
                
                if free < 2.0:  # Less than 2GB free
                    print("⚠️  WARNING: Very low memory available. Consider using CPU for large models.")
                
                del test_tensor
                torch.cuda.empty_cache()
                
                print(f"✅ Using MIG device: {device}")
                return device
                
            except Exception as e:
                print(f"❌ MIG device test failed: {e}")
                return 'cpu'
        else:
            print("❌ No CUDA devices visible in MIG environment")
            return 'cpu'
    
    # Non-MIG environment - use the original logic
    try:
        device = get_gpu_with_max_free_memory()
        print(f"Initial device selection: {device}")
    except (RuntimeError, IndexError) as e:
        print(f"Error in GPU selection: {e}")
        print("Falling back to CPU")
        return 'cpu'
    
    # Validate device selection
    if device.startswith('cuda'):
        device_id = int(device.split(':')[1])
        available_devices = torch.cuda.device_count()
        
        if device_id >= available_devices:
            print(f"Error: Device {device} not available. Available devices: {available_devices}")
            print(f"Available device IDs: {list(range(available_devices))}")
            
            # Find the GPU with maximum free memory among available devices
            max_free_memory = 0
            best_device_id = 0
            
            for i in range(available_devices):
                try:
                    torch.cuda.set_device(i)
                    free_memory = torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_allocated(i)
                    print(f"Device {i}: {free_memory / (1024**3):.2f} GB free memory")
                    
                    if free_memory > max_free_memory:
                        max_free_memory = free_memory
                        best_device_id = i
                except Exception as e:
                    print(f"Error checking device {i}: {e}")
                    continue
            
            device = f'cuda:{best_device_id}'
            device_id = best_device_id
            print(f"Selected device with most free memory: {device} ({max_free_memory / (1024**3):.2f} GB free)")
        
        # Set the device
        try:
            torch.cuda.set_device(device_id)
            print(f"Set CUDA device to: {device_id}")
        except Exception as e:
            print(f"Error setting CUDA device {device_id}: {e}")
            return 'cpu'
        
        # Verify the device is working
        try:
            test_tensor = torch.tensor([1.0], device=device)
            print(f"✅ Device {device} is working correctly")
            return device
        except Exception as e:
            print(f"❌ Device {device} test failed: {e}")
            print("Falling back to CPU")
            return 'cpu'
    
    return device


def get_gpu_with_max_free_memory():
    """
    Returns the GPU device ID with the most available memory.
    Returns 'cpu' if no CUDA devices are available.
    Handles both regular GPUs and MIG instances.
    """
    if not torch.cuda.is_available():
        print("❌ CUDA is not available. No GPUs detected.")
        return 'cpu'
    
    # Check if we're in a MIG environment
    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    is_mig_environment = 'MIG-' in cuda_visible_devices
    
    # First, print basic GPU detection info
    device_count = torch.cuda.device_count()
    print(f"🔍 Detected {device_count} GPU device(s):")
    
    for i in range(device_count):
        props = torch.cuda.get_device_properties(i)
        total_memory_gb = props.total_memory / (1024**3)
        print(f"  GPU {i}: {props.name} ({total_memory_gb:.1f} GB total memory)")
    
    if is_mig_environment:
        print(f"\n🔧 MIG Environment detected: {cuda_visible_devices}")
        print("📊 Using PyTorch memory stats (nvidia-smi may show different info)...")
        
        # In MIG mode, use PyTorch directly since nvidia-smi shows physical GPUs, not MIG slices
        try:
            print("\n💾 MIG Device Memory Usage:")
            print("+" + "-"*70 + "+")
            print(f"| {'GPU':<3} | {'Name':<30} | {'Allocated (GB)':<13} | {'Total (GB)':<10} |")
            print("+" + "-"*70 + "+")
            
            best_device = 0
            max_free = 0
            
            for device_id in range(device_count):
                try:
                    torch.cuda.set_device(device_id)
                    torch.cuda.empty_cache()
                    
                    props = torch.cuda.get_device_properties(device_id)
                    allocated = torch.cuda.memory_allocated(device_id) / (1024**3)
                    total_gb = props.total_memory / (1024**3)
                    free_gb = total_gb - allocated
                    
                    gpu_name = props.name[:30]
                    print(f"| {device_id:<3} | {gpu_name:<30} | {allocated:<13.2f} | {total_gb:<10.1f} |")
                    
                    if free_gb > max_free:
                        max_free = free_gb
                        best_device = device_id
                        
                except Exception as device_error:
                    print(f"| {device_id:<3} | {'Error checking':<30} | {'N/A':<13} | {'N/A':<10} |")
                    continue
            
            print("+" + "-"*70 + "+")
            
            if max_free > 0:
                props = torch.cuda.get_device_properties(best_device)
                print(f"\n✅ Selected MIG device {best_device}: {props.name}")
                print(f"   Available memory: {max_free:.1f} GB")
                return f'cuda:{best_device}'
            else:
                print(f"\n⚠️  Using first available MIG device: cuda:0")
                return 'cuda:0'
                
        except Exception as e:
            print(f"\n❌ MIG device detection failed: {e}")
            return 'cuda:0'
    
    # Non-MIG environment - use nvidia-smi
    try:
        # Try to use nvidia-smi to get detailed memory info
        print("\n📊 Getting detailed memory information via nvidia-smi...")
        
        # Get free memory
        result_free = subprocess.check_output([
            'nvidia-smi', '--query-gpu=memory.free', '--format=csv,nounits,noheader'
        ], encoding='utf-8')
        free_memory = [int(x.strip()) for x in result_free.strip().split('\n')]
        
        # Get used memory
        result_used = subprocess.check_output([
            'nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'
        ], encoding='utf-8')
        used_memory = [int(x.strip()) for x in result_used.strip().split('\n')]
        
        # Get total memory
        result_total = subprocess.check_output([
            'nvidia-smi', '--query-gpu=memory.total', '--format=csv,nounits,noheader'
        ], encoding='utf-8')
        total_memory = [int(x.strip()) for x in result_total.strip().split('\n')]
        
        # Get GPU names
        result_names = subprocess.check_output([
            'nvidia-smi', '--query-gpu=name', '--format=csv,noheader'
        ], encoding='utf-8')
        gpu_names = [name.strip() for name in result_names.strip().split('\n')]
        
        # Filter to only show GPUs that PyTorch can see
        print(f"\n⚠️  Note: nvidia-smi shows {len(free_memory)} GPUs, but PyTorch only sees {device_count}")
        print("Filtering to show only PyTorch-visible devices...")
        
        # Print detailed memory information
        print("\n💾 GPU Memory Usage Details (PyTorch-visible devices):")
        print("+" + "-"*80 + "+")
        print(f"| {'GPU':<3} | {'Name':<25} | {'Used (MB)':<10} | {'Free (MB)':<10} | {'Total (MB)':<11} | {'Usage %':<8} |")
        print("+" + "-"*80 + "+")
        
        max_free_memory = 0
        max_free_device_id = 0
        
        # Only iterate through PyTorch-visible devices
        for i in range(min(device_count, len(free_memory))):
            gpu_name = gpu_names[i][:25] if i < len(gpu_names) else "Unknown"
            used_mb = used_memory[i] if i < len(used_memory) else 0
            free_mb = free_memory[i]
            total_mb = total_memory[i] if i < len(total_memory) else used_mb + free_mb
            usage_percent = (used_mb / total_mb * 100) if total_mb > 0 else 0
            
            print(f"| {i:<3} | {gpu_name:<25} | {used_mb:<10} | {free_mb:<10} | {total_mb:<11} | {usage_percent:<7.1f}% |")
            
            # Find GPU with maximum free memory
            if free_mb > max_free_memory:
                max_free_memory = free_mb
                max_free_device_id = i
        
        print("+" + "-"*80 + "+")
        
        # Check if we found a suitable GPU
        if max_free_memory > 0:
            selected_gpu_name = gpu_names[max_free_device_id] if max_free_device_id < len(gpu_names) else "Unknown"
            print(f"\n✅ Selected GPU {max_free_device_id}: {selected_gpu_name}")
            print(f"   Free memory: {max_free_memory:,} MB ({max_free_memory/1024:.1f} GB)")
            return f'cuda:{max_free_device_id}'
        else:
            print(f"\n⚠️  Using first available GPU: cuda:0")
            return 'cuda:0'
    
    except (subprocess.CalledProcessError, FileNotFoundError, ImportError) as e:
        # If nvidia-smi fails, use PyTorch fallback
        print(f"\n⚠️  nvidia-smi failed ({e}). Using PyTorch memory stats...")
        
        try:
            print("\n💾 GPU Memory Usage (PyTorch estimates):")
            print("+" + "-"*60 + "+")
            print(f"| {'GPU':<3} | {'Name':<25} | {'Allocated (GB)':<13} | {'Total (GB)':<10} |")
            print("+" + "-"*60 + "+")
            
            max_free = 0
            max_device = 0
            
            for device_id in range(device_count):
                try:
                    torch.cuda.set_device(device_id)
                    torch.cuda.empty_cache()
                    
                    props = torch.cuda.get_device_properties(device_id)
                    allocated = torch.cuda.memory_allocated(device_id) / (1024**3)
                    total_gb = props.total_memory / (1024**3)
                    free_estimate = total_gb - allocated
                    
                    gpu_name = props.name[:25]
                    print(f"| {device_id:<3} | {gpu_name:<25} | {allocated:<13.2f} | {total_gb:<10.1f} |")
                    
                    if free_estimate > max_free:
                        max_free = free_estimate
                        max_device = device_id
                        
                except Exception as device_error:
                    print(f"| {device_id:<3} | {'Error checking':<25} | {'N/A':<13} | {'N/A':<10} |")
                    continue
            
            print("+" + "-"*60 + "+")
            
            if max_free > 0:
                props = torch.cuda.get_device_properties(max_device)
                print(f"\n✅ Selected GPU {max_device}: {props.name}")
                print(f"   Estimated free memory: {max_free:.1f} GB")
                return f'cuda:{max_device}'
            else:
                print(f"\n⚠️  Using first available GPU: cuda:0")
                return 'cuda:0'
                
        except Exception as torch_error:
            print(f"\n❌ PyTorch GPU detection also failed: {torch_error}")
            print(f"   Falling back to first GPU: cuda:0")
            return 'cuda:0'