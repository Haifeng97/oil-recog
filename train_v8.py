from ultralytics import YOLO

# 加载预训练模型（可以换成 yolov8s.pt, yolov8m.pt 等）
model = YOLO('yolov8n.pt')

# 训练
model.train(
    data='data.yaml',     # 数据配置文件路径
    epochs=50,
    imgsz=640,
    patience=10,
    batch=16,
    name='oil_detector'
)
