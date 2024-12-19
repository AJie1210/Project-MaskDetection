import torch
from ultralytics import YOLO

# 檢查 PyTorch 是否能使用 GPU
print(torch.cuda.is_available())
print("Current device index:", torch.cuda.current_device())
print("Current device name:", torch.cuda.get_device_name(torch.cuda.current_device()))

# 載入 YOLOv8 模型
model = YOLO('yolov8m.pt')  # 使用 nano 版本進行測試

print(torch.__version__)
print("YOLOv8 模型載入成功")
