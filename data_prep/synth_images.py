import cv2, random
from matplotlib import pyplot as plt
import numpy as np

DIM = (224,224)
DARK_IMAGES_LIST = 'dark_imgs.txt'
BRIGHT_IMAGES_LIST = 'bright_imgs.txt'
BRIGHT_IMAGES_VALS = 'bright_imgs_vals.txt'
ALL_SYNTHS_LIST = 'synth_imgs.txt'
SAVE_PATH = './generated_synths/'
BRIGHTS_PER_DARK = 200
BRIGHTS_PER_BRIGHT = 200


'''
The filtering generated 975 dark images (DI) and 9720 bright ones (BI)
For each image in DI, 200 random images from BI will be chosen as the reflection layer.

Further, the least bright 975 images from BI will be used as background,
and 200 randomly selected images from the brighter half of BI will be used on each of them.

Total examples generated = 2(975*200) = 390000, with dark backgrounds making up 50%
'''

def reshape_img(file_path):
    img = cv2.imread(file_path)
    smaller_dim = min(img.shape[0:2])
    larger_dim = max(img.shape[0:2])
    start_ind = int((larger_dim - smaller_dim)/2)
    img_cropped = img[0:smaller_dim-1, start_ind:start_ind + smaller_dim-1]
    img_scaled = cv2.resize(img_cropped,DIM)
    return img_scaled

def extract_fname(f_path):
    splits = f_path.split('/')
    return (splits[len(splits)-1].split('.'))[0]

def export_synthetic(b,r,b_path,r_path):
    img_b = np.zeros(b.shape,dtype=float)
    img_b += np.array(b)
    img_b /= 255
    r_blur = np.zeros(r.shape,dtype=float)
    r_blur += cv2.GaussianBlur(r,(11,11),1 + 4*np.random.random())
    r_blur /= 255
    i_temp = img_b + r_blur
    exceeds = i_temp[np.where(i_temp>1)]
    if len(exceeds)>0:
        m = np.mean(i_temp[np.where(i_temp>1)])
        r_blur -= 1.3*(m-1)
        r_blur = np.clip(r_blur,0,1)
    img_i = np.zeros(b.shape,dtype=float)
    img_i = img_b + r_blur
    img_i = np.clip(img_i,0,1)
    img_i *= 255

    b_fname = extract_fname(b_path)
    r_fname = extract_fname(r_path)

    img_path = SAVE_PATH + b_fname + "_" + r_fname + ".jpg"
    img_b_path = SAVE_PATH + b_fname + "_" + r_fname + "_b.jpg"
    img_r_path = SAVE_PATH + b_fname + "_" + r_fname + "_r.jpg"

    cv2.imwrite(img_b_path,b)
    cv2.imwrite(img_r_path,r)
    cv2.imwrite(img_path,img_i)

    return img_path

with open(DARK_IMAGES_LIST,'r') as img_list:
    dark_files = np.array(img_list.read().splitlines())

with open(BRIGHT_IMAGES_LIST,'r') as img_list:
    bright_files = np.array(img_list.read().splitlines())

with open(BRIGHT_IMAGES_VALS,'r') as img_list:
    bright_vals = img_list.read().splitlines()
    bright_vals = np.array([float(v) for v in bright_vals])

bright_ordering = np.argsort(bright_vals)

darks = dark_files[np.random.permutation(dark_files.shape[0])]

# print(boundary_ind)
brights_b = bright_files[bright_ordering[bright_files.shape[0] - dark_files.shape[0]:bright_files.shape[0]]]
brights_r = bright_files[bright_ordering[0:bright_files.shape[0]//2]]

total_count = 0
max_count = brights_b.shape[0]*BRIGHTS_PER_BRIGHT + darks.shape[0]*BRIGHTS_PER_DARK
all_files = []

# generating those having dark backgrounds
for dark in darks:
    b = reshape_img(dark)
    rand_inds = random.sample(xrange(0,bright_files.shape[0]),BRIGHTS_PER_DARK)
    for i in range(BRIGHTS_PER_DARK):
        total_count += 1
        r = reshape_img(bright_files[rand_inds[i]])
        fname = export_synthetic(b,r,dark,bright_files[rand_inds[i]])
        print(str.format("{0}/{1}",total_count,max_count) + fname)
        all_files.append(fname)

# generating the rest

for bright in brights_b:
    b = reshape_img(bright)
    rand_inds = random.sample(xrange(0,brights_r.shape[0]),BRIGHTS_PER_BRIGHT)
    for i in range(BRIGHTS_PER_BRIGHT):
        total_count += 1
        r = reshape_img(brights_r[rand_inds[i]])
        fname = export_synthetic(b,r,bright,brights_r[rand_inds[i]])
        print(str.format("{0}/{1}",total_count,max_count) + fname)
        all_files.append(fname)

# write onto text file
with open(ALL_SYNTHS_LIST,"w") as list_file:
    for f in all_files:
        list_file.write(f)
        list_file.write('\n')
