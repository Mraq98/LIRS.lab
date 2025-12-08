import cv2
import time
from ultralytics import YOLO

model = YOLO("runs/classify/train/weights/best.pt")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Ошибка: камера не найдена!")
    exit()

prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Ошибка получения кадра")
        break

    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = model.predict(img_rgb, verbose=False)
    result = results[0]

    cls_id = int(result.probs.top1)
    conf = float(result.probs.top1conf)

    if cls_id == 1:
        text = f"Hand detected ({conf:.2f})"
        color = (0, 255, 0)
    else:
        text = f"No hand ({conf:.2f})"
        color = (0, 0, 255)

    cv2.putText(frame, text, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    cv2.imshow("Hand detection (YOLO)", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
