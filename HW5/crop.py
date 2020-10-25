import cv2
import imutils
import numpy as np
import torchvision
import PIL
import matplotlib.pyplot as plt

def crop_imgs(set_name, add_pixels_value=0):
    """
    Finds the extreme points on the image and crops the rectangular out of them
    """
    set_new = []
    for img in set_name:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # threshold the image, then perform a series of erosions +
        # dilations to remove any small regions of noise
        thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)

        # find contours in thresholded image, then grab the largest one
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)

        # find the extreme points
        extLeft = tuple(c[c[:, :, 0].argmin()][0])
        extRight = tuple(c[c[:, :, 0].argmax()][0])
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        extBot = tuple(c[c[:, :, 1].argmax()][0])

        ADD_PIXELS = add_pixels_value
        new_img = img[extTop[1]-ADD_PIXELS:extBot[1]+ADD_PIXELS, extLeft[0]-ADD_PIXELS:extRight[0]+ADD_PIXELS].copy()
        set_new.append(new_img)

    return np.array(set_new)

# No_img = []
# Yes_img = []
# # img=mpimg.imread('image_name.png')
# for i in range(1, 100) :
#         img = cv2.imread('tumor_dataset/no/'+str(i)+' no.jpg')
#         if img is  None:
#             continue
#         else:
#             No_img.append(img)
#
# for i in range(259) :
#     img = cv2.imread('tumor_dataset/yes/Y' + str(i) + '.jpg')
#     if img is None:
#         continue
#     else:
#         Yes_img.append(img)
#
# print(len(No_img))
# print(len(Yes_img))
#
# new_image_No = crop_imgs(No_img,0)
# new_image_Yes = crop_imgs(Yes_img)
#
# for n,img in enumerate(new_image_No) :
#     cv2.imwrite("tumor_dataset/no_new_new/no"+str(n)+".jpg",img)
#
# for n,img in enumerate(new_image_Yes) :
#     cv2.imwrite("tumor_dataset/yes_new/yes"+str(n)+".jpg",img)
#
#
# transforms = torchvision.transforms.Compose([
#     # torchvision.transforms.RandomCrop(110, padding=4),
#     torchvision.transforms.Pad(50),
#     torchvision.transforms.ToPILImage(),
#     # torchvision.transforms.ColorJitter(0.4,0.6,0,0),
#     torchvision.transforms.Resize((64,64)),
#     torchvision.transforms.RandomHorizontalFlip(),
#     torchvision.transforms.RandomRotation(10, resample=PIL.Image.BILINEAR)
#
#
# ])
# #
# #
# dataset = torchvision.datasets.ImageFolder('data', transform=transforms)
# for n, i in enumerate(dataset) :
#     i[0].save("no/no"+str(n+480)+".jpg")
#
#
#
# # print(len(dataset))
#
# # print(type(dataset))
# #
# # d = dataset[0]
# # print(d[0])
# # d[0].show()
# # print(len(d))
