import cv2
import numpy as np


def nothing(x):
    pass



cv2.namedWindow('Calibration')


cv2.createTrackbar('H_min', 'Calibration', 0, 179, nothing)
cv2.createTrackbar('H_max', 'Calibration', 179, 179, nothing)
cv2.createTrackbar('S_min', 'Calibration', 0, 255, nothing)
cv2.createTrackbar('S_max', 'Calibration', 255, 255, nothing)
cv2.createTrackbar('V_min', 'Calibration', 0, 255, nothing)
cv2.createTrackbar('V_max', 'Calibration', 255, 255, nothing)

cap = cv2.VideoCapture(0)

print("Настройте trackbars чтобы ваш объект был БЕЛЫМ на черном фоне")
print("Нажмите 'q' для выхода")

while True:
    ret, frame = cap.read()
    if not ret:
        break


    h_min = cv2.getTrackbarPos('H_min', 'Calibration')
    h_max = cv2.getTrackbarPos('H_max', 'Calibration')
    s_min = cv2.getTrackbarPos('S_min', 'Calibration')
    s_max = cv2.getTrackbarPos('S_max', 'Calibration')
    v_min = cv2.getTrackbarPos('V_min', 'Calibration')
    v_max = cv2.getTrackbarPos('V_max', 'Calibration')


    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_color = np.array([h_min, s_min, v_min])
    upper_color = np.array([h_max, s_max, v_max])
    mask = cv2.inRange(hsv, lower_color, upper_color)


    cv2.imshow('Original', frame)
    cv2.imshow('Mask', mask)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print(f"Ваши настройки HSV:")
print(f"Lower: [{h_min}, {s_min}, {v_min}]")
print(f"Upper: [{h_max}, {s_max}, {v_max}]")