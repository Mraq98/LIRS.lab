from ultralytics import YOLO

model = YOLO('runs/classify/train/weights/best.pt')

results = model('test.png')
print(f"Класс: {results[0].names[results[0].probs.top1]}")
print(f"Уверенность: {results[0].probs.top1conf:.3f}")