import cv2 as cv
import numpy as np

def write_image(path, images, extension):
  for i in range(0, len(images)):
    cv.imwrite(path + str(i + 1) + '.' + extension,images[i])