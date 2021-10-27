import cv2 as cv
import numpy as np
import math
from show_image import show_image
from arnold_transform import arnold_transform
from write_image import write_image

def transform_image_to_bits(img):
  array = img.flatten()
  bytes = array.tobytes()
  return bin(int.from_bytes(bytes, byteorder='big'))[2:] 

def transform_bits_to_image(bit_array, size):
  bytes =  int(bit_array, 2).to_bytes((len(bit_array) + 7) // 8, 'big')
  buf = np.frombuffer(bytes, dtype=np.uint8)
  img = buf.reshape(size,size)
  return img

def calculate_dc(img, originx, originy, pixelblocklength):
  count = 0
  for i in range(0 ,pixelblocklength):
    for j in range(0 ,pixelblocklength):
      count += img[originx + i][originy + j]
  return count/pixelblocklength

def apply_dc(img, originx, originy, pixelblocklength, dc_dif):
  newimage = img.copy()
  for i in range(0 ,pixelblocklength):
    for j in range(0 ,pixelblocklength):
      value = (dc_dif)/pixelblocklength
      if(dc_dif < 0):
        value = math.floor(value)
      else:
        value = math.ceil(value)
      newimage[originx + i][originy + j] += value
  return newimage

def embed_binary(img, size, pixel_block_size, bit_array):
  newimage = img.copy()
  bit_count = 0
  for i in range(0 ,int(size/pixel_block_size)):
    if(bit_count >= len(bit_array)):
      break
    for j in range(0 ,int(size/(pixel_block_size * 2))):
      if(bit_count >= len(bit_array)):
        break
      dc1 = calculate_dc(newimage,i * pixel_block_size,j * (pixel_block_size * 2),pixel_block_size) 
      dc2 = calculate_dc(newimage,i * pixel_block_size,j * (pixel_block_size * 2) + pixel_block_size,pixel_block_size) 
      avg = ( dc1 + dc2 )/ 2
      dc1_n = dc1
      dc2_n = dc2
      if bit_array[bit_count] == '0' and (dc1 - dc2) > 0:
        dc1_n = avg - 0.5
        dc2_n = avg + 0.5
      elif bit_array[bit_count] == '1' and (dc1 - dc2) <= 0:
        dc1_n = avg + 0.5
        dc2_n = avg - 0.5
      newimage = apply_dc(newimage,i * pixel_block_size,j * (pixel_block_size * 2),pixel_block_size, dc1_n - dc1) 
      newimage = apply_dc(newimage,i * pixel_block_size,j * (pixel_block_size * 2) + pixel_block_size, pixel_block_size, dc2_n - dc2)
      bit_count += 1 
  return newimage

def unembed_binary(img, size, pixel_block_size, bit_array_size):
  newimage = img.copy()
  bit_array = ''
  for i in range(0 ,int(size/pixel_block_size)):
    if(len(bit_array) >= bit_array_size):
      break
    for j in range(0 ,int(size/(pixel_block_size * 2))):
      if(len(bit_array) >= bit_array_size):
        break
      dc1 = calculate_dc(newimage,i * pixel_block_size,j * (pixel_block_size * 2),pixel_block_size) 
      dc2 = calculate_dc(newimage,i * pixel_block_size,j * (pixel_block_size * 2) + pixel_block_size,pixel_block_size) 
      if (dc1 - dc2) > 0:
        bit_array += '1'
      else:
        bit_array += '0'
  return bit_array


def perform_watermark_embedding(img,img_size,logo,logo_size,pixel_block_size):

  # transforma a imagem em uma sequencia de bits
  logo_bit_array = transform_image_to_bits(logo)

  # da o embed na imagem original com a logomarca
  embedded_image = embed_binary(img, img_size, pixel_block_size, logo_bit_array)

  # remove a sequencia de bits da logomarca
  new_bit_array = unembed_binary(embedded_image, img_size, pixel_block_size, len(logo_bit_array))

  # transforma a sequencia de bits em imagem
  new_scrambled_logo = transform_bits_to_image(new_bit_array, logo_size)

  return (embedded_image,new_scrambled_logo)


def main(imagepath,logopath,resultspath):
  img_size = 512
  logo_size = 128
  pixel_block_size = 1
  img = cv.imread(imagepath,cv.IMREAD_COLOR)
  img = cv.resize(img,(img_size,img_size),interpolation = cv.INTER_AREA)

  logo = cv.imread(logopath,cv.IMREAD_COLOR)
  logo = cv.resize(logo,(logo_size,logo_size),interpolation = cv.INTER_AREA)

  img_b,img_g,img_r = cv.split(img)
  logo_b,logo_g,logo_r = cv.split(logo)

  # revisar periodo da transformação de arnold pelo tamanho da imagem
  enciter = int(logo_size/2) # qualquer numero entre 0 e o periodo
  deciter = (int(logo_size/16)) * 12 - enciter # o periodo menos o enciter
  scrambled_logo_r = arnold_transform(logo_r, enciter)
  scrambled_logo_g = arnold_transform(logo_g, enciter)
  scrambled_logo_b = arnold_transform(logo_b, enciter)

  (embedded_image_r, scrambled_logo_r) = perform_watermark_embedding(img_r, img_size, scrambled_logo_r, logo_size, pixel_block_size)
  (embedded_image_g, scrambled_logo_g) = perform_watermark_embedding(img_g, img_size, scrambled_logo_g, logo_size, pixel_block_size)
  (embedded_image_b, scrambled_logo_b) = perform_watermark_embedding(img_b, img_size, scrambled_logo_b, logo_size, pixel_block_size)

  #unscramble_logo
  unscrambled_logo_r = arnold_transform(scrambled_logo_r, deciter)
  unscrambled_logo_g = arnold_transform(scrambled_logo_g, deciter)
  unscrambled_logo_b = arnold_transform(scrambled_logo_b, deciter)

  embedded_image = cv.merge((embedded_image_b, embedded_image_g, embedded_image_r))
  scrambled_logo = cv.merge((scrambled_logo_b, scrambled_logo_g, scrambled_logo_r))
  unscrambled_logo = cv.merge((unscrambled_logo_b, unscrambled_logo_g, unscrambled_logo_r))

  # Imagem diferença 
  diff_image = cv.absdiff(img, embedded_image)
  diff_perc_img = (np.count_nonzero(diff_image) / (img_size*img_size * 3)) * 100
  print('Diff Image: ', diff_perc_img, '%')

  # Logo diferença
  diff_logo = cv.absdiff(logo, unscrambled_logo)
  diff_perc_logo = (np.count_nonzero(diff_logo) / (logo_size*logo_size * 3)) * 100
  print('Diff Logo: ' ,diff_perc_logo, '%')

  write_image('results/' + resultspath,[img, logo,embedded_image,scrambled_logo, unscrambled_logo ,diff_image,diff_logo],'png')
  # show_image([img, logo, embedded_image,scrambled_logo, unscrambled_logo,diff_image,diff_logo])
  return [diff_perc_img, diff_perc_logo]

logos = ['assets/logos/peugeot.jpg',
         'assets/logos/coca-cola.jfif']
images = ['assets/images/lena.tif',
          'assets/images/mandril_color.tif',
          'assets/images/peppers_color.tif',
          'assets/images/fruits.png',
          'assets/images/HappyFish.jpg',
          'assets/images/tulips.png',]

avg_img_diff = 0
avg_logo_diff = 0
for i in range(0 ,len(images)):
  for j in range(0 ,len(logos)):
    res = main(images[i],logos[j],str(j) + '_' + str(i) + '_')
    avg_img_diff += res[0]
    avg_logo_diff += res[1]
number_of_tests = len(images) * len(logos)

print(avg_img_diff/number_of_tests)
print(avg_logo_diff/number_of_tests)