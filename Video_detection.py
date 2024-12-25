import cv2
from ultralytics import YOLO

# 載入訓練好的 YOLOv8 模型
model = YOLO('C:\\Users\\Wayne\\Documents\\GitHub\\Project-MaskDetection\\mask_detection_results\\mask_detection_run-100epochs1-L\\weights\\best.pt')  # 替換為您的模型路徑

# 指定影片檔案的路徑
video_path = 'C:\\Users\\Wayne\\Videos\\Captures\\Japan.mp4'  # 替換為您的影片檔案路徑

# 開啟影片檔案
cap = cv2.VideoCapture(video_path)

# 檢查影片是否成功開啟
if not cap.isOpened():
    print(f"無法開啟影片檔案: {video_path}")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("影片播放結束或無法讀取影片幀")
        break

    # 使用 YOLOv8 模型進行預測
    results = model(frame, conf=0.5)  # conf 設定信心門檻

    # 繪製預測結果
    annotated_frame = results[0].plot()

    # 顯示影像
    cv2.imshow('Mask Detection', annotated_frame)

    # 按 'q' 鍵退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 釋放資源
cap.release()
cv2.destroyAllWindows()
