# In the name of Allah
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

img = cv2.imread('files/IMG_20230224_184809.jpg', 0)

blur = cv2.medianBlur(img, 101)

edges = cv2.Canny(blur, 10, 10)

mask = np.zeros(img.shape, np.uint8)

ps = np.array([[p0, p1] for p0, p1 in zip(np.where(edges == 255)[1], np.where(edges == 255)[0])])
ps = ps[ps[:, 0] > 1150]
ps = ps[ps[:, 0] < 2950]
ps = ps[ps[:, 1] > 650]
ps = ps[ps[:, 1] < 2450]

cv2.fillPoly(mask, [ps], 255)

mask = cv2.medianBlur(mask, 181)

print(Counter(mask.ravel())[255])

plt.imshow(mask, cmap='gray')
plt.show()