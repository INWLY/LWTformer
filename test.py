import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
# Check GPU devices (optional)
print(f"Number of GPUs: {torch.cuda.device_count()}")
print(f"Current GPU: {torch.cuda.get_device_name(0)}")