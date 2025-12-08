from ultralytics import YOLO

model = YOLO('yolo11s-cls.pt')

model.train(
    data='cls_dataset',
    epochs=60,
    imgsz=256,
    batch=32,
    device='cpu',
    lr0=0.001,
    augment=True
)

results = model('test.png')
print(f"Класс: {results[0].names[results[0].probs.top1]}")
print(f"Уверенность: {results[0].probs.top1conf:.3f}")
