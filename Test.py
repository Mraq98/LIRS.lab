import cv2
import numpy as np



def nothing(x):
    pass



cv2.namedWindow('HSV Color Filter', cv2.WINDOW_NORMAL)
cv2.resizeWindow('HSV Color Filter', 1920, 1200)


cv2.createTrackbar('H_min', 'HSV Color Filter', 0, 179, nothing)
cv2.createTrackbar('H_max', 'HSV Color Filter', 179, 179, nothing)
cv2.createTrackbar('S_min', 'HSV Color Filter', 0, 255, nothing)
cv2.createTrackbar('S_max', 'HSV Color Filter', 255, 255, nothing)
cv2.createTrackbar('V_min', 'HSV Color Filter', 0, 255, nothing)
cv2.createTrackbar('V_max', 'HSV Color Filter', 255, 255, nothing)


cv2.setTrackbarPos('H_min', 'HSV Color Filter', 0)
cv2.setTrackbarPos('H_max', 'HSV Color Filter', 179)
cv2.setTrackbarPos('S_min', 'HSV Color Filter', 0)
cv2.setTrackbarPos('S_max', 'HSV Color Filter', 255)
cv2.setTrackbarPos('V_min', 'HSV Color Filter', 0)
cv2.setTrackbarPos('V_max', 'HSV Color Filter', 255)


cap = cv2.VideoCapture(0)


cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:

    ret, frame = cap.read()
    if not ret:
        break


    frame_resized = cv2.resize(frame, (426, 320))


    rgb_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)


    hsv_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2HSV)


    h_min = cv2.getTrackbarPos('H_min', 'HSV Color Filter')
    h_max = cv2.getTrackbarPos('H_max', 'HSV Color Filter')
    s_min = cv2.getTrackbarPos('S_min', 'HSV Color Filter')
    s_max = cv2.getTrackbarPos('S_max', 'HSV Color Filter')
    v_min = cv2.getTrackbarPos('V_min', 'HSV Color Filter')
    v_max = cv2.getTrackbarPos('V_max', 'HSV Color Filter')


    lower_bound = np.array([h_min, s_min, v_min])
    upper_bound = np.array([h_max, s_max, v_max])


    mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)


    result_bgr = cv2.bitwise_and(frame_resized, frame_resized, mask=mask)


    result_rgb = cv2.bitwise_and(rgb_frame, rgb_frame, mask=mask)


    main_canvas = np.zeros((900, 1600, 3), dtype=np.uint8)


    main_canvas[50:370, 50:476] = frame_resized
    cv2.putText(main_canvas, 'BGR Original', (50, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


    main_canvas[50:370, 562:988] = rgb_frame
    cv2.putText(main_canvas, 'RGB Converted', (562, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


    main_canvas[420:740, 50:476] = hsv_frame
    cv2.putText(main_canvas, 'HSV Converted', (50, 410),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    main_canvas[420:740, 562:988] = mask_bgr
    cv2.putText(main_canvas, 'Mask', (562, 410),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


    main_canvas[50:370, 1074:1500] = result_bgr
    cv2.putText(main_canvas, 'BGR Result', (1074, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


    main_canvas[420:740, 1074:1500] = result_rgb
    cv2.putText(main_canvas, 'RGB Result', (1074, 410),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


    info_text = f'H: [{h_min}-{h_max}]  S: [{s_min}-{s_max}]  V: [{v_min}-{v_max}]'
    cv2.putText(main_canvas, info_text, (500, 800),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)


    cv2.imshow('HSV Color Filter', main_canvas)


    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break


cap.release()
cv2.destroyAllWindows()