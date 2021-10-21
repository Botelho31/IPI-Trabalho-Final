import cv2 as cv
import numpy as np
from show_image import show_image

def arnold_transform(img, enciter):
  tempImg = np.copy(img)
  temp = np.zeros((len(img),len(img[0])), dtype=np.uint8)
  m = len(img)
  for iter in range(0 ,enciter):
    for i in range(0 ,m):
      for j in range(0 ,m):
        x = (i + j) % m
        y = (i + 2 * j) % m
        temp[x][y] = tempImg[i][j]
    tempImg = np.copy(temp)
    temp = np.zeros((len(img),len(img[0])), dtype=np.uint8)
  return tempImg