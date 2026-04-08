import numpy as np
import matplotlib.pyplot as plt
from kernels import vertical_line, horizontal_line, diagonal_line

a = np.zeros((50, 50))
a[20:45, 25] = 1
a[10, 2:30] = 1

d = np.zeros((20, 20))
for k in range(20):
    d[k,k] = 1
a[0:20, 30:50] = d

v = vertical_line(a)
h = horizontal_line(a)
d = diagonal_line(a)

#angle picture
H, W = 50, 50
angle = np.zeros((H, W))
m = 0.3
b = 30

for x in range(W):
    y = int(m * x + b)
    if 0 <= y < H:
        angle[y, x] = 1
#angle kernels
Av = vertical_line(angle)
Ah = horizontal_line(angle)
Ad = diagonal_line(angle)

#a plot and kernels
plt.subplot(1,4,1)
plt.title("Original")
plt.imshow(a, cmap='gray')

plt.subplot(1,4,2)
plt.title("Vertical")
plt.imshow(v, cmap='gray')

plt.subplot(1,4,3)
plt.title("Horizontal")
plt.imshow(h, cmap='gray')

plt.subplot(1,4,4)
plt.title("Diagonal")
plt.imshow(d, cmap='gray')

plt.show()

#plot angle and angle kernels
plt.subplot(1,4,1)
plt.title("Original angle pic")
plt.imshow(angle, cmap='gray')

plt.subplot(1,4,2)
plt.title("Vertical")
plt.imshow(Av, cmap='gray')

plt.subplot(1,4,3)
plt.title("Horizontal")
plt.imshow(Ah, cmap='gray')

plt.subplot(1,4,4)
plt.title("Diagonal")
plt.imshow(Ad, cmap='gray')

plt.show()