import cv2
import numpy as np
import math


class ObjectDetector:
    def __init__(self):
        self.lower_color = np.array([100, 75, 151])
        self.upper_color = np.array([179, 255, 255])

    def detect_object(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_color, self.upper_color)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None, None, None, None, None, None

        largest_contour = max(contours, key=cv2.contourArea)

        if cv2.contourArea(largest_contour) < 500:
            return None, None, None, None, None, None

        M = cv2.moments(largest_contour)
        if M["m00"] == 0:
            return None, None, None, None, None, None

        center_x = int(M["m10"] / M["m00"])
        center_y = int(M["m01"] / M["m00"])
        center = (center_x, center_y)

        rect = cv2.minAreaRect(largest_contour)
        angle = rect[2]

        width, height = rect[1]
        if width < height:
            minor_axis_length = width
        else:
            minor_axis_length = height

        line_length = minor_axis_length / 2

        angle_rad = math.radians(angle)
        end_x1 = int(center_x + line_length * math.cos(angle_rad))
        end_y1 = int(center_y + line_length * math.sin(angle_rad))
        end_x2 = int(center_x - line_length * math.cos(angle_rad))
        end_y2 = int(center_y - line_length * math.sin(angle_rad))

        capture_line = [(end_x1, end_y1), (end_x2, end_y2)]

        angle_horizontal, angle_vertical = self.calculate_angles(capture_line)

        return largest_contour, center, capture_line, angle_horizontal, angle_vertical, mask

    def calculate_angles(self, line):
        x1, y1 = line[0]
        x2, y2 = line[1]

        dx = x2 - x1
        dy = y2 - y1

        angle_horizontal = math.degrees(math.atan2(dy, dx))
        if angle_horizontal < 0:
            angle_horizontal += 180

        angle_vertical = 90 - angle_horizontal
        if angle_vertical < 0:
            angle_vertical += 180

        return angle_horizontal, angle_vertical

    def draw_results(self, frame, contour, center, capture_line, angle_horizontal, angle_vertical, mask):
        result_frame = frame.copy()

        cv2.drawContours(result_frame, [contour], -1, (0, 255, 0), 2)

        cv2.circle(result_frame, center, 5, (255, 0, 0), -1)

        cv2.line(result_frame, capture_line[0], capture_line[1], (0, 0, 255), 2)

        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        cv2.drawContours(result_frame, [box], 0, (255, 255, 0), 1)

        info_text = [
            f"Center: ({center[0]}, {center[1]})",
            f"Angle H: {angle_horizontal:.1f}°",
            f"Angle V: {angle_vertical:.1f}°"
        ]

        for i, text in enumerate(info_text):
            cv2.putText(result_frame, text, (10, 30 + i * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return result_frame


def main():
    detector = ObjectDetector()
    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Запуск детекции объекта. Нажмите 'q' для выхода")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Ошибка захвата видео")
            break

        contour, center, capture_line, angle_h, angle_v, mask = detector.detect_object(frame)

        if contour is not None:
            result_frame = detector.draw_results(frame, contour, center, capture_line, angle_h, angle_v, mask)

            cv2.imshow('Color Mask', mask)
        else:
            result_frame = frame
            mask_display = np.zeros_like(frame[:, :, 0])
            cv2.imshow('Color Mask', mask_display)

        cv2.imshow('Object Detection', result_frame)

        # Обработка клавиш
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break


    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()