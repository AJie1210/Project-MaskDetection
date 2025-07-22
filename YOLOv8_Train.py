from ultralytics import YOLO

def main():
    model = YOLO('yolov8m.pt')

    model.train(
        data='C:\\Users\\ytes6\\OneDrive\\文件\\GitHub\\Project-MaskDetection\\Mask Detection.v3i.yolov8\\data.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        name='mask_detection_run-100epochs1-m',
        project='mask_detection_results',
        patience=20,                     # 早停耐心值
        lr0=0.001,                        # 初始學習率
        lrf=0.01,                        # 最終學習率
        momentum=0.9,                     # 動量
        weight_decay=0.0001,             # 權重衰減
        optimizer='AdamW',               # 優化器
        augment=True,                    # 啟用數據增強
        device=0,                        # 指定 GPU（0 為預設）
        workers=8,                       # 數據加載工人數量
        verbose=True,                  # 詳細輸出
    )

if __name__ == '__main__':
    main()
