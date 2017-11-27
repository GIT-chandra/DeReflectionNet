import glob, cv2
from matplotlib import pyplot as plt
import numpy as np


IMG_PATH = './JPEGImages/*'
DARK_IMAGES_LIST = 'dark_imgs.txt'
BRIGHT_IMAGES_LIST = 'bright_imgs.txt'
BRIGHT_IMAGES_VALS = 'bright_imgs_vals.txt'
BRIGHT_LIM = 0.1
DARK_LIM = 0.4
DARK_PIX_BOUND = 35  # for pixel values between 0 and 255

def get_darkpix(img_file):
    img = cv2.imread(img_file)
    r_dark = (img[:,:,0] < DARK_PIX_BOUND)
    g_dark = (img[:,:,1] < DARK_PIX_BOUND)
    b_dark = (img[:,:,2] < DARK_PIX_BOUND)
    return float((r_dark*g_dark*b_dark).sum())/(img[:,:,0].flatten().shape[0])

all_img_paths = glob.glob(IMG_PATH)
darks = []
brights = []
brights_vals = []
for path in all_img_paths:
    print(path)
    dark_fraction = get_darkpix(path)
    print(dark_fraction)
    if dark_fraction > DARK_LIM:
        darks.append(path)

        # display them
        # img = cv2.imread(path)
        # img_corrected = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # plt.imshow(img_corrected)
        # plt.show()
    elif dark_fraction < BRIGHT_LIM:
        brights.append(path)
        brights_vals.append(dark_fraction)

# write onto text file
with open(DARK_IMAGES_LIST,"w") as list_file:
    for dark in darks:
        list_file.write(dark)
        list_file.write('\n')

with open(BRIGHT_IMAGES_LIST,"w") as list_file:
    for bright in brights:
        list_file.write(bright)
        list_file.write('\n')

with open(BRIGHT_IMAGES_VALS,"w") as list_file:
    for bv in brights_vals:
        list_file.write(str(bv))
        list_file.write('\n')
