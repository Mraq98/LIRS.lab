import cv2
import numpy as np
import glob

camera_matrix = np.array([
    [3030.09, 0, 1887.83],
    [0, 3038.98, 2244.69],
    [0, 0, 1]
], dtype=np.float32)

dist_coeffs = np.zeros((5, 1), dtype=np.float32)

marker_size = 0.095

marker_3d = np.array([
    [-marker_size/2,  marker_size/2, 0],
    [ marker_size/2,  marker_size/2, 0],
    [ marker_size/2, -marker_size/2, 0],
    [-marker_size/2, -marker_size/2, 0],
], dtype=np.float32)

image_paths = sorted(glob.glob("aruco_images/*.jpg"))
print("Найдено изображений:", len(image_paths))

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(dictionary, parameters)

print("\nРезультаты:\n")
print("№\tИмя файла\t\tРасстояние по ArUco (м)")

for i, img_path in enumerate(image_paths, 1):
    img = cv2.imread(img_path)

    corners, ids, rejected = detector.detectMarkers(img)

    if ids is None:
        print(f"{i}\t{img_path}\tМАРКЕР НЕ НАЙДЕН")
        continue

    c = corners[0].reshape(-1, 2)

    success, rvec, tvec = cv2.solvePnP(
        marker_3d,
        c,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_IPPE_SQUARE
    )

    if not success:
        print(f"{i}\t{img_path}\tОшибка solvePnP")
        continue

    distance = np.linalg.norm(tvec)

    print(f"{i}\t{img_path}\t{distance:.3f} м")
