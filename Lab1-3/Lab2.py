import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('gosling.jpeg')

if image is None:
    print("Ошибка: Не удалось загрузить изображение 'gosling.jpeg'")
    exit()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(gray, 50, 150)

inverted_edges = 255 - edges

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.title('Оригинальное изображение')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title('Фильтр Canny (границы)')
plt.imshow(edges, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title('Графический рисунок')
plt.imshow(inverted_edges, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()