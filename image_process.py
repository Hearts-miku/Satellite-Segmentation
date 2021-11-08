import matplotlib.pyplot as plt
import os
import numpy as np
import cv2
import PIL.Image as Image


image1_names = os.listdir('Data/val/images')
image2_names = os.listdir('Data/val/images0')

im1 = Image.open('Data/val/images/'+image1_names[12])
im1_G = im1.convert('L')

im2 = Image.open('Data/val/images0/'+image2_names[0])
im2_G = im2.convert('L')

arr1 = np.array(im1, dtype=np.uint8)
arr1_G = np.array(im1_G, dtype=np.uint8)
d1 = arr1.reshape(262144,3)
d1_G = arr1_G.reshape(262144,1)

arr2 = np.array(im2, dtype=np.uint8)
arr2_G = np.array(im2_G, dtype=np.uint8)
d2 = arr2.reshape(262144,3)
d2_G = arr2_G.reshape(262144,1)



print(np.min(arr1, axis=(0,1)))
print(np.min(arr2, axis=(0,1)))
print(np.min(arr1_G, axis=(0,1)))
print(np.min(arr2_G, axis=(0,1)))

print(np.max(arr1, axis=(0,1)))
print(np.max(arr2, axis=(0,1)))
print(np.max(arr1_G, axis=(0,1)))
print(np.max(arr2_G, axis=(0,1)))
'''
plt.subplot(1, 2, 1)
plt.imshow(arr1_G, cmap='gray')
plt.subplot(1, 2, 2)
plt.imshow(arr2_G, cmap='gray')
'''
plt.subplot(1, 2, 1)
plt.boxplot(d1_G)
plt.subplot(1, 2, 2)
plt.boxplot(d2_G)
plt.savefig('g_box')

