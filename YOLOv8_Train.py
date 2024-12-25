from ultralytics import YOLO

def main():
    # 載入預訓練模型
    model = YOLO('yolov8l.pt')  # 或 'yolov8l.pt' 根據需求選擇

    # 開始訓練
    model.train(
        data='data.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        name='mask_detection_run-100epochs1-L',
        project='mask_detection_results'
    )

if __name__ == '__main__':
    main()
