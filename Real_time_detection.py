import cv2
from ultralytics import YOLO

# 載入訓練好的 YOLOv8 模型
model = YOLO('C:\\Users\\Wayne\\Documents\\GitHub\\Project-MaskDetection\\mask_detection_results\\mask_detection_run-50epochs1-m\\weights\\best.pt')  # 替換為您的模型路徑

# 開啟攝影機（0 為預設攝影機）
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("無法讀取攝影機畫面")
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
