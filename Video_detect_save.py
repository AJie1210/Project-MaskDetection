import cv2
from ultralytics import YOLO
from datetime import datetime
import csv

# 載入 YOLOv8 模型
model = YOLO(r'C:\Users\ytes6\OneDrive\文件\GitHub\Project-MaskDetection\mask_detection_results\mask_detection_run-100epochs1-m\weights\best.pt')

# 指定影片路徑
video_path = r'C:\Users\ytes6\Videos\Captures\videoplayback (2).mp4'
cap = cv2.VideoCapture(video_path)

# 建立 CSV 紀錄檔案
csv_path = 'tracking_mask_log.csv'
csv_file = open(csv_path, 'w', newline='', encoding='utf-8')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['datetime', 'track_id', 'mask_status', 'confidence'])

# 用來避免重複統計同一人
logged_ids = {}

# 啟用 YOLO 內建追蹤器（ByteTrack）
model.trackers = 'bytetrack.yaml'

# 取得影片 FPS
fps = cap.get(cv2.CAP_PROP_FPS)
fps_text = f"FPS: {fps:.2f}"

class_map = {
    0: "incorrect_mask",
    1: "masked",
    2: "no_mask"
}

color_map = {
    0: (255, 0, 0),
    1: (0, 255, 0),
    2: (0, 0, 255)
}


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 使用追蹤模式推論
    results = model.track(source=frame, persist=True, conf=0.5)

    # 繪製推論畫面
    annotated = results[0].plot()

    # 判斷是否有框
    if results[0].boxes.id is not None:
        for box in results[0].boxes:
            track_id = int(box.id[0])
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # 類別轉換文字與顏色
            label = class_map.get(cls_id, "unknown")
            color = color_map.get(cls_id, (255, 255, 255))

            # 顯示 ID 與分類
            x, y = int(box.xyxy[0][0]), int(box.xyxy[0][1])
            cv2.putText(annotated, f"ID:{track_id} {label}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            # 僅記錄第一次出現的 ID
            if track_id not in logged_ids:
                logged_ids[track_id] = cls_id
                csv_writer.writerow([now, track_id, label, f"{conf:.2f}"])

    # 顯示 FPS
    cv2.putText(annotated, fps_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)

    # 顯示畫面
    cv2.imshow("Mask Tracking", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
csv_file.close()
cv2.destroyAllWindows()
print(f"結果已儲存至 {csv_path}")