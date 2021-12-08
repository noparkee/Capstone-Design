import numpy as np
import os
import PIL
from PIL import Image, ImageFilter

import matplotlib.pyplot as plt

import pathlib
import glob



def crop_image(img_file):
  fn = img_file.filename
  img_width, img_height = img_file.size
  
  if img_width != img_height:
    m = min(img_width, img_height)
    img = img_file.crop((img_width//2 - m//2 , img_height//2 - m//2, img_width//2 + m//2 , img_height//2 + m//2))
    img_width = m
    img_height = m
    img.save(fn[:-4]+'-cropcenter.jpg')

  else:
    # 좌상
    img = img_file.crop((0, 0, img_width//2+10, img_height//2+10))
    img = img.resize((img_width, img_height))
    img.save(fn[:-4]+'-cropleftup.jpg')

    # 우상
    img = img_file.crop((img_width//2-10, 0, img_width, img_height//2+10))
    img = img.resize((img_width, img_height))
    img.save(fn[:-4]+'-croprightup.jpg')

    # 좌하
    img = img_file.crop((0, img_height//2-10, img_width//2+10, img_height))
    img = img.resize((img_width, img_height))
    img.save(fn[:-4]+'-cropleftdown.jpg')

    # 우하
    img = img_file.crop((img_width//2-10, img_height//2-10, img_width, img_height))
    img = img.resize((img_width, img_height))
    img.save(fn[:-4]+'-croprightdown.jpg')

    # 중앙
    img = img_file.crop((img_width//4, img_height//4, img_width - img_width//4, img_height - img_height//4))
    img = img.resize((img_width, img_height))
    img.save(fn[:-4]+'-cropcenter.jpg')
    
def rotate_image(img_file):
  fn = img_file.filename
  img_width, img_height = img_file.size

  for r in range(45, 360, 45):
    if r % 10 != 0:   # 45도 단위
      img = img_file.rotate(r)
      img = img.crop((img_width//6, img_height//6, img_width - img_width//6, img_height - img_height//6))
      img = img.resize((img_width, img_height))

    else:
      img = img_file.rotate(90)

    img.save(fn[:-4]+'-rotate'+str(r)+'.jpg')

def blur_image(img_file):
  fn = img_file.filename
  img_width, img_height = img_file.size

  img = img_file.filter(ImageFilter.GaussianBlur(10))

  img.save(fn[:-4]+'-blur.jpg')
  


data_dir = '../../data'

# original
paper = ['1000won', '5000won', '10000won', '50000won']
coin = ['10won', '50won', '100won', '500won']

for p in paper:       # 지폐에 대해서
  paper_lst = os.listdir(str(data_dir)+"/"+p)
  for pl in paper_lst:
    image = Image.open(str(data_dir)+"/"+p+"/"+pl)
    crop_image(image)
    rotate_image(image)
    blur_image(image)
  print("Done Folder :" + p)
print("Done Paper Augmentation...")

for c in coin:
  coin_lst = os.listdir(str(data_dir)+"/"+c)
  for cl in coin_lst:
    image = Image.open(str(data_dir)+"/"+c+"/"+cl)
    crop_image(image)
    rotate_image(image)
    blur_image(image)
  print("Done Folder :" + c)
print("Done Coin Augmentation...")

