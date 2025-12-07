import os
import shutil
import random
from pathlib import Path


def create_binary_classification_dataset(source_folder, output_folder, val_ratio=0.2):
    for split in ['train', 'val']:
        for class_name in ['0', '1']:
            Path(f"{output_folder}/{split}/{class_name}").mkdir(parents=True, exist_ok=True)

    for class_name in ['0', '1']:

        class_path = os.path.join(source_folder, class_name)
        if not os.path.exists(class_path):
            print(f"Папка {class_path} не найдена!")
            continue

        images = [f for f in os.listdir(class_path)
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        random.shuffle(images)
        split_idx = int(len(images) * (1 - val_ratio))

        train_images = images[:split_idx]
        val_images = images[split_idx:]

        for img in train_images:
            shutil.copy2(os.path.join(class_path, img),
                         os.path.join(output_folder, 'train', class_name, img))

        for img in val_images:
            shutil.copy2(os.path.join(class_path, img),
                         os.path.join(output_folder, 'val', class_name, img))

        print(f"Класс {class_name}: {len(train_images)} train, {len(val_images)} val")


create_binary_classification_dataset(
    "hand_dataset",  # <-- твоя папка
    "cls_dataset"  # <-- выходная папка
)
