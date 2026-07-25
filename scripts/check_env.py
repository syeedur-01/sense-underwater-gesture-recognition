import torch
import platform
import sys


def main():
    print(f"Python: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA capability: {torch.cuda.get_device_capability(0)}")
        print(f"VRAM (GB): {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f}")
    else:
        print("WARNING: CUDA not available — training will run on CPU (not viable for this project).")


if __name__ == "__main__":
    main()
