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
        lr0=0.01,                     # 初始學習率
        lrf=0.01,                     # 最終學習率
        momentum=0.937,               # 動量
        weight_decay=0.0005,          # 權重衰減
        optimizer='AdamW',            # 優化器
        patience=15,                  # 早停耐心值
        augment=True,                 # 啟用數據增強
        workers=8,                    # 資料加載器工作數
        cache=True,                   # 啟用緩存
        device=0,                     # 使用第一個 GPU
        verbose=True,                 # 啟用詳細輸出
        freeze=None,                  # 不凍結任何層
    )

if __name__ == '__main__':
    main()
