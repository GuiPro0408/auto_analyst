
import torch
try:
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device capability: {torch.cuda.get_device_capability(0)}")
        print(f"Arch list: {torch.cuda.get_arch_list()}")
except Exception as e:
    print(f"Error: {e}")
