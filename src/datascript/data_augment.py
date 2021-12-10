import os
import numpy as np
import argparse
from PIL import Image, ImageFilter



parser = argparse.ArgumentParser()
parser.add_argument('--type', type=str)
args = parser.parse_args()
TYPE = args.type


def crop_image(img_file, only_center):
  fn = img_file.filename
  img_width, img_height = img_file.size
  
  if only_center:
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

if TYPE == 'others':
  # for "others" folder
  others_lst = os.listdir(str(data_dir)+"/others")
  for ol in others_lst:
    image = Image.open(str(data_dir)+"/others"+"/"+ol)
    print(ol)
    crop_image(image)
    rotate_image(image)
    blur_image(image)
else:
  # original
  paper = ['1000won', '5000won', '10000won', '50000won']
  coin = ['10won', '50won', '100won', '500won']

  for p in paper:       # 지폐에 대해서
    paper_lst = os.listdir(str(data_dir)+"/"+p)
    for pl in paper_lst:
      image = Image.open(str(data_dir)+"/"+p+"/"+pl)
      crop_image(image, False)
      rotate_image(image)
      blur_image(image)
    print("Done Folder :" + p)
  print("Done Paper Augmentation...")

  for c in coin:      # 동전에 대해서
    coin_lst = os.listdir(str(data_dir)+"/"+c)
    for cl in coin_lst:
      image = Image.open(str(data_dir)+"/"+c+"/"+cl)
      crop_image(image, True)
      rotate_image(image)
      blur_image(image)
    print("Done Folder :" + c)
  print("Done Coin Augmentation...")
