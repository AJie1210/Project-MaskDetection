from ultralytics import YOLO

def main():
    # 載入預訓練模型
    model = YOLO('yolov8l.pt')  # 或 'yolov8l.pt' 根據需求選擇

    # 開始訓練
    model.train(
        data='data.yaml',
        epochs=150,
        imgsz=640,
        batch=16,
        name='mask_detection_run-150epochs1-L',
        project='mask_detection_results',
        patience=15,                     # 早停耐心值
        lr0=0.01,                        # 初始學習率
        lrf=0.01,                        # 最終學習率
        momentum=0.937,                  # 動量
        weight_decay=0.0005,             # 權重衰減
        optimizer='AdamW',               # 優化器
        augment=True,                    # 啟用數據增強
        device=0,                        # 指定 GPU（0 為預設）
        workers=8,                       # 數據加載工人數量
    )

if __name__ == '__main__':
    main()
