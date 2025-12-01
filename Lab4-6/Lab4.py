import cv2
import glob
import numpy as np
import os


def resize_image_for_detection(img, max_width=1200):
    if img.shape[1] > max_width:
        scale = max_width / img.shape[1]
        new_width = max_width
        new_height = int(img.shape[0] * scale)
        return cv2.resize(img, (new_width, new_height))
    return img


def check_images_in_folder(folder_path):
    print("🔍 Поиск изображений в папке...")

    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    all_images = []

    for ext in extensions:
        pattern = os.path.join(folder_path, ext)
        images = glob.glob(pattern)
        all_images.extend(images)
        if images:
            print(f"Найдено {len(images)} файлов {ext}")

    if not all_images:
        print("❌ В папке не найдено изображений!")
        return []

    print(f"\n📊 Всего найдено {len(all_images)} изображений:")
    for img_path in all_images:
        print(f"  - {os.path.basename(img_path)}")

    return all_images


def test_chessboard_detection(image_path, chessboard_sizes):
    print(f"\n🔎 Тестирование: {os.path.basename(image_path)}")

    img = cv2.imread(image_path)
    if img is None:
        print("  ❌ Не удалось загрузить изображение")
        return None

    print(f"  📐 Оригинальный размер: {img.shape}")

    img_small = resize_image_for_detection(img)
    print(f"  📐 Размер для обработки: {img_small.shape}")

    gray = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)

    found = False
    for size in chessboard_sizes:
        flags = (cv2.CALIB_CB_ADAPTIVE_THRESH +
                 cv2.CALIB_CB_NORMALIZE_IMAGE +
                 cv2.CALIB_CB_FILTER_QUADS)

        ret, corners = cv2.findChessboardCorners(gray, size, flags)

        if ret:
            print(f"  ✅ Найдены углы для размера {size}")
            found = True
            return size

    if not found:
        print("  ❌ Углы не найдены ни для одного размера")

    return None


def calibrate_camera_with_detected_size(images, chessboard_size, square_size=0.03):

    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objp *= square_size

    objpoints = []
    imgpoints = []
    successful_images = []

    print(f"\n🎯 Калибровка с размером доски {chessboard_size}")

    for i, image_path in enumerate(images):
        print(f"  Обработка {i + 1}/{len(images)}: {os.path.basename(image_path)}")

        img = cv2.imread(image_path)
        if img is None:
            print("    ❌ Не удалось загрузить изображение")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        flags = (cv2.CALIB_CB_ADAPTIVE_THRESH +
                 cv2.CALIB_CB_NORMALIZE_IMAGE +
                 cv2.CALIB_CB_FILTER_QUADS)

        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, flags)

        if ret:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.0001)
            corners_refined = cv2.cornerSubPix(gray, corners, (25, 25), (-1, -1), criteria)

            objpoints.append(objp)
            imgpoints.append(corners_refined)
            successful_images.append(image_path)

            print(f"    ✅ Углы найдены и уточнены")
        else:
            print(f"    ❌ Углы не найдены")

    print(f"\n📊 Успешно обработано: {len(successful_images)}/{len(images)} изображений")

    if len(successful_images) < 5:
        print("❌ Недостаточно изображений для калибровки (нужно минимум 5)")
        return None, None

    # Выполняем калибровку
    print("🔧 Выполняем калибровку...")
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )

    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error

    print(f"📏 Средняя ошибка репроекции: {mean_error / len(objpoints):.3f} пикселей")

    return mtx, dist


def main():
    print("Лабораторная работа №4: Калибровка камеры")
    print("=" * 60)

    CHESSBOARD_IMAGES_PATH = "chessboard_images"
    SQUARE_SIZE = 0.03

    if not os.path.exists(CHESSBOARD_IMAGES_PATH):
        print(f"❌ Папка '{CHESSBOARD_IMAGES_PATH}' не существует!")
        print("Создайте папку 'chessboard_images' в той же директории, где находится скрипт")
        return

    images = check_images_in_folder(CHESSBOARD_IMAGES_PATH)
    if not images:
        return

    chessboard_sizes_to_try = [
        (9, 6),
        (8, 5),
        (7, 7),
        (8, 6),
        (6, 4),
        (5, 4),
        (10, 7),
    ]

    print(f"\n🧪 Тестируем {len(chessboard_sizes_to_try)} разных размеров доски...")

    working_size = None
    test_images = images[:3]

    for image_path in test_images:
        working_size = test_chessboard_detection(image_path, chessboard_sizes_to_try)
        if working_size:
            print(f"\n🎉 Найден рабочий размер: {working_size}")
            break

    if not working_size:
        print("\n❌ Не удалось найти подходящий размер доски!")
        return

    mtx, dist = calibrate_camera_with_detected_size(images, working_size, SQUARE_SIZE)

    if mtx is not None:
        print("\n" + "=" * 60)
        print("✅ КАЛИБРОВКА ЗАВЕРШЕНА УСПЕШНО!")
        print("=" * 60)

        print("\n📊 МАТРИЦА КАМЕРЫ:")
        print("[[fx,  0, cx],")
        print(" [ 0, fy, cy],")
        print(" [ 0,  0,  1]]")

        print(f"\n🎯 ФОКУСНЫЕ РАССТОЯНИЯ (для отчета):")
        print(f"fx = {mtx[0, 0]:.2f} пикселей")
        print(f"fy = {mtx[1, 1]:.2f} пикселей")

        print(f"\n📍 ЦЕНТР ИЗОБРАЖЕНИЯ:")
        print(f"cx = {mtx[0, 2]:.2f} пикселей")
        print(f"cy = {mtx[1, 2]:.2f} пикселей")

        print(f"\n📐 ПАРАМЕТРЫ КАЛИБРОВКИ:")
        print(f"Размер доски: {working_size} внутренних углов")
        print(f"Размер квадрата: {SQUARE_SIZE * 100} см")
        print(f"Количество изображений: {len(images)}")

        # Сохраняем результаты
        np.savez("camera_calibration.npz",
                 camera_matrix=mtx,
                 distortion_coefficients=dist,
                 chessboard_size=working_size)
        print(f"\n💾 Результаты сохранены в 'camera_calibration.npz'")

        print("\n📝 ДЛЯ ОТЧЕТА ЛАБОРАТОРНОЙ РАБОТЫ №4:")
        print("=" * 40)
        print(f"Матрица камеры:")
        print(f"[[{mtx[0, 0]:.2f}, 0, {mtx[0, 2]:.2f}]")
        print(f" [0, {mtx[1, 1]:.2f}, {mtx[1, 2]:.2f}]")
        print(f" [0, 0, 1]]")
        print(f"\nФокусные расстояния:")
        print(f"fx = {mtx[0, 0]:.2f} пикселей")
        print(f"fy = {mtx[1, 1]:.2f} пикселей")

    else:
        print("\n❌ Калибровка не удалась!")


if __name__ == "__main__":
    main()