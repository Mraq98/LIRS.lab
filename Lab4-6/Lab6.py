import sys
import os
import cv2
import torch
import numpy as np

# =============================
# 1. Подключаем библиотеку Depth-Anything-V2
# =============================
project_path = r"C:\Users\kirya\Documents\GitHub\LIRS.lab\Depth-Anything-V2"
sys.path.append(project_path)

from depth_anything_v2.dpt import DepthAnythingV2

# =============================
# 2. Загружаем модель Depth-Anything-V2 Small (vits)
# =============================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model_configs = {
    'vits': {
        'encoder': 'vits',
        'features': 64,
        'out_channels': [48, 96, 192, 384]
    }
}

model = DepthAnythingV2(**model_configs['vits'])

ckpt_path = os.path.join(project_path, "checkpoints", "depth_anything_v2_vits.pth")
state = torch.load(ckpt_path, map_location="cpu")
model.load_state_dict(state)

model = model.to(DEVICE).eval()

print("Модель успешно загружена!")

# =============================
# 3. Загружаем изображение
# =============================
image_path = r"C:\Users\kirya\Documents\GitHub\LIRS.lab\Lab4-6\lab6_images\1.jpg"
img = cv2.imread(image_path)

if img is None:
    raise FileNotFoundError("Изображение не найдено!")

# Для вывода и обработки нужен RGB
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# =============================
# 4. Получение КАРТЫ ОТНОСИТЕЛЬНОЙ ГЛУБИНЫ
# =============================
with torch.no_grad():
    rel_depth = model.infer_image(img_rgb)

# Нормализуем карту глубины для отображения
rel_depth_norm = (rel_depth - rel_depth.min()) / (rel_depth.max() - rel_depth.min())
rel_depth_vis = (rel_depth_norm * 255).astype(np.uint8)
cv2.imwrite("relative_depth.png", rel_depth_vis)

print("Относительная карта глубины сохранена → relative_depth.png")

# =============================
# 5. КАЛИБРОВКА Depth Anything по данным OpenCV
# =============================

# Параметры камеры из калибровки
camera_matrix = np.array([
    [3030.09, 0, 1887.83],
    [0, 3038.98, 2244.69],
    [0, 0, 1]
], dtype=np.float32)

dist_coeffs = np.zeros((5, 1), dtype=np.float32)
marker_size = 0.095  # размер маркера в метрах

# 3D модель маркера
marker_3d = np.array([
    [-marker_size / 2, marker_size / 2, 0],
    [marker_size / 2, marker_size / 2, 0],
    [marker_size / 2, -marker_size / 2, 0],
    [-marker_size / 2, -marker_size / 2, 0],
], dtype=np.float32)

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters()

detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
corners, ids, _ = detector.detectMarkers(img)

if ids is None:
    raise Exception("На изображении не найдено ArUco маркеров!")

print(f"Найдено маркеров: {len(ids)}")

# Определяем точные расстояния OpenCV
opencv_distances = []
centers = []

for i, corner in enumerate(corners):
    c = corner[0]
    cx = int(c[:, 0].mean())
    cy = int(c[:, 1].mean())
    centers.append((cx, cy))

    success, rvec, tvec = cv2.solvePnP(
        marker_3d,
        c.astype(np.float32),
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_IPPE_SQUARE
    )

    if success:
        distance = np.linalg.norm(tvec)
        opencv_distances.append(distance)
        print(f"Маркер {i + 1} - OpenCV: {distance:.3f} м")


# Калибруем Depth Anything по данным OpenCV
def calibrate_depth_anything(rel_depth, centers, opencv_distances):
    """
    Калибруем Depth Anything используя точные измерения OpenCV
    """
    # Получаем относительные значения Depth Anything в центрах маркеров
    da_relative_values = []
    for (cx, cy) in centers:
        y_start = max(0, cy - 2)
        y_end = min(rel_depth.shape[0], cy + 3)
        x_start = max(0, cx - 2)
        x_end = min(rel_depth.shape[1], cx + 3)
        rel_val = np.mean(rel_depth[y_start:y_end, x_start:x_end])
        da_relative_values.append(rel_val)

    # Находим коэффициенты линейной калибровки
    da_values = np.array(da_relative_values)
    cv_values = np.array(opencv_distances)

    # Линейная регрессия: cv_distance = a * da_relative + b
    A = np.column_stack([da_values, np.ones(len(da_values))])
    a, b = np.linalg.lstsq(A, cv_values, rcond=None)[0]

    print(f"Калибровочные коэффициенты: a = {a:.3f}, b = {b:.3f}")

    # Применяем калибровку ко всей карте глубины
    calibrated_depth = a * rel_depth + b

    return calibrated_depth


# Калиброванная метрическая глубина
metric_depth_calibrated = calibrate_depth_anything(rel_depth, centers, opencv_distances)

# =============================
# 6. Измеряем КАЛИБРОВАННЫЕ расстояния Depth Anything
# =============================

metric_distances_calibrated = []

for i, (cx, cy) in enumerate(centers):
    y_start = max(0, cy - 2)
    y_end = min(metric_depth_calibrated.shape[0], cy + 3)
    x_start = max(0, cx - 2)
    x_end = min(metric_depth_calibrated.shape[1], cx + 3)

    depth_val = np.mean(metric_depth_calibrated[y_start:y_end, x_start:x_end])
    metric_distances_calibrated.append(depth_val)

print("\nКАЛИБРОВАННЫЕ расстояния Depth Anything (в метрах):")
for i, d in enumerate(metric_distances_calibrated):
    print(f"Маркер {i + 1}: {d:.3f} м")

# =============================
# 7. Сохраняем калиброванную карту глубины
# =============================

# Визуализация калиброванной карты
md_vis_calibrated = (metric_depth_calibrated / metric_depth_calibrated.max() * 255).astype(np.uint8)
cv2.imwrite("metric_depth_calibrated.png", md_vis_calibrated)

# Цветная версия
metric_norm_calibrated = (metric_depth_calibrated - metric_depth_calibrated.min()) / (
        metric_depth_calibrated.max() - metric_depth_calibrated.min()
)
metric_u8_calibrated = (metric_norm_calibrated * 255).astype("uint8")
metric_color_calibrated = cv2.applyColorMap(metric_u8_calibrated, cv2.COLORMAP_TURBO)
cv2.imwrite("metric_depth_colored_calibrated.png", metric_color_calibrated)


# =============================
# 8. Сравнение результатов
# =============================

def ratio(d1, d2):
    return d1 / d2 if d2 != 0 else float('inf')


print("\nСравнение отношений расстояний:")

if len(metric_distances_calibrated) >= 3 and len(opencv_distances) >= 3:
    # Отношения для калиброванного Depth Anything
    metric_ratio = ratio(metric_distances_calibrated[0], metric_distances_calibrated[1])
    metric_ratio2 = ratio(metric_distances_calibrated[1], metric_distances_calibrated[2])

    # Отношения для OpenCV
    opencv_ratio = ratio(opencv_distances[0], opencv_distances[1])
    opencv_ratio2 = ratio(opencv_distances[1], opencv_distances[2])

    print(f"Depth Anything (калибр.): d1/d2 = {metric_ratio:.3f}, d2/d3 = {metric_ratio2:.3f}")
    print(f"OpenCV:                    d1/d2 = {opencv_ratio:.3f}, d2/d3 = {opencv_ratio2:.3f}")

# =============================
# 9. Итоговые результаты для таблицы
# =============================

print("\n" + "=" * 50)
print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ ДЛЯ ТАБЛИЦЫ")
print("=" * 50)

print("\nАбсолютные расстояния:")
print("№ маркера | Depth Anything (калибр.), м | OpenCV, м")
print("----------|-----------------------------|-----------")
for i in range(min(len(metric_distances_calibrated), len(opencv_distances))):
    diff = abs(metric_distances_calibrated[i] - opencv_distances[i])
    print(f"    {i + 1}     |           {metric_distances_calibrated[i]:.3f}          |   {opencv_distances[i]:.3f}")

if len(metric_distances_calibrated) >= 3 and len(opencv_distances) >= 3:
    print("\nОтношения расстояний:")
    print("Отношение | Depth Anything (калибр.) | OpenCV")
    print("----------|--------------------------|--------")
    print(f"  d1/d2   |         {metric_ratio:.3f}         | {opencv_ratio:.3f}")
    print(f"  d2/d3   |         {metric_ratio2:.3f}         | {opencv_ratio2:.3f}")

print("\nГотово!")
print("Файлы сохранены:")
print(" - relative_depth.png — относительная карта глубины")
print(" - metric_depth_calibrated.png — калиброванная метрическая карта глубины")
print(" - metric_depth_colored_calibrated.png — цветная калиброванная карта глубины")