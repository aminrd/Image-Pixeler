import numpy as np
import cv2
import os
import sys

# Global variables:
accept_list = ['.png', '.jpg', '.jpeg', '.bmp']
#===================

def valid_ext(fname):
    ext = os.path.splitext(fname)[-1]
    if ext in accept_list:
        return True
    return False

def convert_to_square(image):
    if image.shape[0] < image.shape[1]:
        s = image.shape[0]
        i0 = (image.shape[1]-s)//2
        i1 = i0+s
        return image[:, i0:i1,...]
    else:
        s = image.shape[1]
        i0 = (image.shape[0]-s)//2
        i1 = i0+s
        return image[i0:i1,:,...]

def convert(main_image, im_size):
    sq = convert_to_square(main_image)
    return cv2.resize(sq, (im_size, im_size))

def Importer(input_dir, output_dir, recursion=True, im_size=512, Gray=False):
    gallery = []

    pnumber = 0

    if recursion==False:
        for file in os.listdir(input_dir):

            if not valid_ext(file):
                continue

            fname = os.path.join(input_dir, file)
            if Gray:
                img = cv2.imread(fname, 0)
            else:
                img = cv2.imread(fname, 1)

            img = convert(img, im_size)

            pname = os.path.join(output_dir, 'pixel{}.jpg'.format(pnumber))
            pnumber += 1
            cv2.imwrite(pname, img)
            gallery += [img]
    else:
        for root, subdirs, files in os.walk(input_dir):
            for file in files:

                if not valid_ext(file):
                    continue

                fname = os.path.join(root, file)
                if Gray:
                    img = cv2.imread(fname, 0)
                else:
                    img = cv2.imread(fname, 1)

                img = convert(img, im_size)

                pname = os.path.join(output_dir, 'pixel{}.jpg'.format(pnumber))
                pnumber += 1
                cv2.imwrite(pname, img)
                gallery += [img]
    return gallery

