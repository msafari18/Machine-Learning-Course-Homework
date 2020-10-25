import crop
import cv2
import torchvision
import PIL
import numpy as np
from random import shuffle
class dataset() :

   def __init__(self):
      self.No_img = []
      self.Yes_img = []
      self.No_crop = []
      self.Yes_crop = []

   def read_data(self):
      for i in range(1, 100):
         img = cv2.imread('tumor_dataset/no/' + str(i) + ' no.jpg')
         if img is None:
            continue
         else:
            self.No_img.append(img)

      for i in range(259):
         img = cv2.imread('tumor_dataset/yes/Y' + str(i) + '.jpg')
         if img is None:
            continue
         else:
            self.Yes_img.append(img)

   def crop_image(self):
      self.No_crop = [PIL.Image.fromarray(i) for i in crop.crop_imgs(self.No_img)]
      self.Yes_crop = [PIL.Image.fromarray(i) for i in crop.crop_imgs(self.Yes_img)]

   def tranform_for_agumantation(self):
      transforms_1 = torchvision.transforms.Compose([
         torchvision.transforms.Pad(50),
         torchvision.transforms.Resize((64, 64)),
         torchvision.transforms.RandomHorizontalFlip(),
         torchvision.transforms.RandomRotation(10, resample=PIL.Image.BILINEAR)])

      transforms_2 = torchvision.transforms.Compose([
         torchvision.transforms.Resize((64, 64)),
         torchvision.transforms.RandomHorizontalFlip(),
         torchvision.transforms.RandomRotation(100, resample=PIL.Image.BILINEAR)])

      transforms_3 = torchvision.transforms.Compose([
         torchvision.transforms.Resize((64, 64)),
         torchvision.transforms.RandomHorizontalFlip(),
         torchvision.transforms.RandomRotation(160, resample=PIL.Image.BILINEAR)
      ])

      transforms_4 = torchvision.transforms.Compose([
         torchvision.transforms.Resize((64, 64)),
         torchvision.transforms.RandomHorizontalFlip()])

      transforms_5 = torchvision.transforms.Compose([
         torchvision.transforms.Pad(50),
         torchvision.transforms.Resize((64, 64)),
         torchvision.transforms.RandomHorizontalFlip(),
         torchvision.transforms.RandomRotation(100, resample=PIL.Image.BILINEAR)])

      new_yes = []
      new_no = []
      for i in self.Yes_crop :
         new_yes.append(transforms_1(i))
         new_yes.append(transforms_2(i))
         new_yes.append(transforms_3(i))
         new_yes.append(transforms_4(i))

      new_yes = [np.array(pic) for pic in new_yes]


      for i in self.No_crop :
         new_no.append(transforms_1(i))
         new_no.append(transforms_2(i))
         new_no.append(transforms_3(i))
         new_no.append(transforms_4(i))
         new_no.append(transforms_5(i))
         new_no.append(transforms_2(i))

      new_no = [np.array(pic) for pic in new_no]
      shuffle(new_no)
      shuffle(new_yes)

      return new_yes, new_no

data = dataset()
data.read_data()
data.crop_image()
yes_img, no_img = data.tranform_for_agumantation()