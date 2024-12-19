import torch
from ultralytics import YOLO

print("PyTorch version:", torch.__version__)
print("CUDA version:", torch.version.cuda)
print("CUDNN version:", torch.backends.cudnn.version())
print("Is CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0))
else:
    print("No GPU detected.")
