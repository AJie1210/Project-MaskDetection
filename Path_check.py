import os

train_path = r'C:\Users\Wayne\Documents\GitHub\Project-MaskDetection\mask-detection\train\images'
val_path = r'C:\Users\Wayne\Documents\GitHub\Project-MaskDetection\mask-detection\valid\images'

print("訓練集圖片是否存在:", os.path.exists(train_path))
print("驗證集圖片是否存在:", os.path.exists(val_path))

# Best module : YOLOv8-L 100 epochs
