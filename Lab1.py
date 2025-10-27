import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('gosling.jpeg')

if image is None:
    print("Ошибка: Не удалось загрузить изображение 'gosling.jpeg'")
    print("Убедитесь, что файл находится в правильной директории")
    exit()

median_filtered = cv2.medianBlur(image, 5)

blurred = cv2.GaussianBlur(image, (15, 15), 0)

kernel = np.array([[0, -1, 0],
                   [-1, 5, -1],
                   [0, -1, 0]])
sharpened_conv = cv2.filter2D(image, -1, kernel)

blurred_for_sharp = cv2.GaussianBlur(image, (5, 5), 0)
sharpened_unsharp = cv2.addWeighted(image, 1.5, blurred_for_sharp, -0.5, 0)

sharpened = sharpened_unsharp

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)

sobel_combined = cv2.magnitude(sobelx, sobely)

edges = cv2.convertScaleAbs(sobel_combined)

edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

custom_kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
custom_filtered = cv2.filter2D(image, -1, custom_kernel)

combined1 = cv2.addWeighted(blurred, 0.7, edges_colored, 0.3, 0)

combined_final = cv2.addWeighted(combined1, 0.6, sharpened, 0.4, 0)

def show_all_results(original, median, gaussian_blur, sobel_edges,
                    sharp_conv, sharp_unsharp, custom, combined):
    plt.figure(figsize=(16, 12))

    plt.subplot(3, 3, 1)
    plt.title('Оригинальное изображение')
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(3, 3, 2)
    plt.title('Медианный фильтр')
    plt.imshow(cv2.cvtColor(median, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(3, 3, 3)
    plt.title('Гауссово размытие')
    plt.imshow(cv2.cvtColor(gaussian_blur, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(3, 3, 4)
    plt.title('Границы (Собель)')
    plt.imshow(cv2.cvtColor(sobel_edges, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(3, 3, 5)
    plt.title('Резкость (свертка)')
    plt.imshow(cv2.cvtColor(sharp_conv, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(3, 3, 6)
    plt.title('Резкость (маска нерезкости)')
    plt.imshow(cv2.cvtColor(sharp_unsharp, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(3, 3, 7)
    plt.title('Собственный фильтр')
    plt.imshow(cv2.cvtColor(custom, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(3, 3, 8)
    plt.title('Комбинированный результат')
    plt.imshow(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(3, 3, 9)
    plt.text(0.1, 0.9, 'Использованные фильтры:', fontsize=12, fontweight='bold')
    plt.text(0.1, 0.7, '1. Медианный фильтр (5x5)', fontsize=10)
    plt.text(0.1, 0.6, '2. Гауссово размытие (15x15)', fontsize=10)
    plt.text(0.1, 0.5, '3. Оператор Собеля (5x5)', fontsize=10)
    plt.text(0.1, 0.4, '4. Маска нерезкости', fontsize=10)
    plt.text(0.1, 0.3, '5. Собственный фильтр', fontsize=10)
    plt.text(0.1, 0.2, '6. Комбинирование', fontsize=10)
    plt.axis('off')

    plt.tight_layout()
    plt.show()

show_all_results(image, median_filtered, blurred, edges_colored,
                sharpened_conv, sharpened_unsharp, custom_filtered, combined_final)

def show_main_results(original, blurred, edges, sharpened, combined):
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 3, 1)
    plt.title('Оригинальное изображение')
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.title('Размытие по Гауссу')
    plt.imshow(cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.title('Выделение границ (Собель)')
    plt.imshow(cv2.cvtColor(edges, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(2, 3, 4)
    plt.title('Повышение резкости')
    plt.imshow(cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(2, 3, 5)
    plt.title('Комбинация изображений')
    plt.imshow(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.tight_layout()
    plt.show()

print("Отображаются основные результаты...")
show_main_results(image, blurred, edges_colored, sharpened, combined_final)