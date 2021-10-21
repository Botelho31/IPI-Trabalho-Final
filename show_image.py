import cv2 as cv
import numpy as np

def show_image(images):
  for i in range(0, len(images)):
    cv.imshow('image {}'.format(i), images[i])
  cv.waitKey(0)
  cv.destroyAllWindows()