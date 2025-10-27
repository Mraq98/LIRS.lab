import cv2
import time
from PIL import ImageFont, ImageDraw, Image
import numpy as np

# === Загрузка каскадов Хаара ===
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

# === Видеопоток ===
cap = cv2.VideoCapture(0)

prev_time = 0
fps_values = []  # список для хранения всех FPS

# === Настройка шрифта (путь к .ttf файлу) ===
font_path = "C:/Windows/Fonts/arial.ttf"  # Windows
font = ImageFont.truetype(font_path, 28)

def draw_text_pil(img, text, position, font, color=(255, 255, 0)):
    """Функция для рисования текста с поддержкой кириллицы."""
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Не удалось получить изображение с камеры")
        break

    # === Расчёт FPS ===
    current_time = time.time()
    time_diff = current_time - prev_time
    fps = 1 / time_diff if time_diff > 0 else 0
    prev_time = current_time
    fps_values.append(fps)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    message = ""

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=10)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.7, minNeighbors=22)
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 255, 255), 2)

        # === Проверка состояний ===
        if len(eyes) < 2:
            message = "Открой глаза"
        elif len(smiles) == 0:
            message = "Улыбнись"
        else:
            message = "Отлично!"

    # === Отображение FPS ===
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # === Сообщение пользователю ===
    if message:
        frame = draw_text_pil(frame, message, (50, 60), font, color=(0, 255, 255))

    cv2.imshow('Распознавание лица', frame)

    # ESC — выход
    if cv2.waitKey(1) == 27:
        break

# === Расчёт среднего FPS после выхода ===
if fps_values:
    avg_fps = sum(fps_values) / len(fps_values)
    print(f"Среднее значение FPS за время работы: {avg_fps:.2f}")

cap.release()
cv2.destroyAllWindows()
