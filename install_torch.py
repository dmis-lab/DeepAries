#!/usr/bin/env python3
"""
PyTorch installation script for DeepAries project.
Automatically detects CUDA availability and installs the appropriate PyTorch version.
"""
import subprocess
import sys
import platform

def check_cuda_available():
    """Check if CUDA is available on the system."""
    try:
        import torch
        if torch.cuda.is_available():
            cuda_version = torch.version.cuda
            print(f"CUDA is available: {cuda_version}")
            return True, cuda_version
        else:
            print("CUDA is not available. Installing CPU version.")
            return False, None
    except ImportError:
        # Try to detect CUDA without torch
        try:
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
            if result.returncode == 0:
                print("NVIDIA GPU detected. Installing GPU version.")
                # Default to CUDA 12.1 if detected
                return True, "12.1"
            else:
                print("No NVIDIA GPU detected. Installing CPU version.")
                return False, None
        except FileNotFoundError:
            print("nvidia-smi not found. Installing CPU version.")
            return False, None

def get_pytorch_index_url(cuda_version=None):
    """Get the PyTorch index URL based on CUDA version."""
    if cuda_version is None:
        # CPU version
        return "https://download.pytorch.org/whl/cpu"
    
    # Map CUDA versions to PyTorch index URLs
    cuda_major = cuda_version.split('.')[0] if '.' in cuda_version else cuda_version
    
    if cuda_major == "12":
        # Use CUDA 12.1 (most common)
        return "https://download.pytorch.org/whl/cu121"
    elif cuda_major == "11":
        # Use CUDA 11.8 (most common for 11.x)
        return "https://download.pytorch.org/whl/cu118"
    else:
        # Default to CUDA 12.1
        print(f"Unknown CUDA version {cuda_version}, defaulting to CUDA 12.1")
        return "https://download.pytorch.org/whl/cu121"

def install_pytorch(use_gpu=False, cuda_version=None):
    """Install PyTorch with appropriate version."""
    packages = ["torch", "torchvision", "torchaudio"]
    
    if use_gpu:
        index_url = get_pytorch_index_url(cuda_version)
        print(f"Installing PyTorch GPU version from {index_url}")
        cmd = [
            sys.executable, "-m", "pip", "install",
            *packages,
            "--index-url", index_url
        ]
    else:
        print("Installing PyTorch CPU version")
        index_url = get_pytorch_index_url(None)
        cmd = [
            sys.executable, "-m", "pip", "install",
            *packages,
            "--index-url", index_url
        ]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=True)
    return result.returncode == 0

def main():
    """Main installation function."""
    print("=" * 60)
    print("PyTorch Installation Script for DeepAries")
    print("=" * 60)
    
    # Check if user wants to force CPU or GPU
    force_cpu = "--cpu" in sys.argv
    force_gpu = "--gpu" in sys.argv
    
    if force_cpu:
        print("Forcing CPU installation (--cpu flag detected)")
        use_gpu = False
        cuda_version = None
    elif force_gpu:
        print("Forcing GPU installation (--gpu flag detected)")
        use_gpu = True
        cuda_version = "12.1"  # Default CUDA version
    else:
        # Auto-detect
        use_gpu, cuda_version = check_cuda_available()
    
    # Install PyTorch
    try:
        success = install_pytorch(use_gpu, cuda_version)
        if success:
            print("\n" + "=" * 60)
            print("PyTorch installation completed successfully!")
            print("=" * 60)
            
            # Verify installation
            try:
                import torch
                print(f"\nPyTorch version: {torch.__version__}")
                if use_gpu:
                    print(f"CUDA available: {torch.cuda.is_available()}")
                    if torch.cuda.is_available():
                        print(f"CUDA version: {torch.version.cuda}")
                        print(f"GPU device: {torch.cuda.get_device_name(0)}")
                else:
                    print("Using CPU version")
            except ImportError:
                print("Warning: Could not import torch after installation")
        else:
            print("PyTorch installation failed.")
            sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"Error installing PyTorch: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

