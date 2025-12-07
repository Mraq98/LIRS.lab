from ultralytics import YOLO

# Загружаем классификационную модель
model = YOLO('yolo11s-cls.pt')

model.train(
    data='cls_dataset',     # Папка с train/val
    epochs=60,              # Рекомендуемые эпохи
    imgsz=256,              # Картинка побольше — лучше обучение
    batch=32,
    device='cpu',           # Можешь сменить на 'cuda' если есть
    lr0=0.001,
    augment=True            # ВАЖНО: включаем аугментации
)

# Проверка после обучения
results = model('test.png')
print(f"Класс: {results[0].names[results[0].probs.top1]}")
print(f"Уверенность: {results[0].probs.top1conf:.3f}")
