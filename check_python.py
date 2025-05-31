import sys
import torch

print("=== Python Version ===")
print(sys.version)
print("\n=== PyTorch Info ===")
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)
    print("Current device:", torch.cuda.current_device())
    print("Device name:", torch.cuda.get_device_name(0))
    print("Memory allocated: {:.2f} MB".format(torch.cuda.memory_allocated() / 1024**2))
    print("Memory reserved: {:.2f} MB".format(torch.cuda.memory_reserved() / 1024**2))
else:
    print("CUDA is not available. Please check your installation.")
