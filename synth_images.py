import cv2
from matplotlib import pyplot as plt
import numpy as np

DIM = (224,224)
DARK_IMAGES_LIST = 'dark_imgs.txt'
BRIGHT_IMAGES_LIST = 'bright_imgs.txt'
BRIGHT_IMAGES_VALS = 'bright_imgs_vals.txt'
ALL_SYNTHS_LIST = 'synth_imgs.txt'
SAVE_PATH = './generated_synths/'
BRIGHTS_PER_DARK = 9
BRIGHTS_PER_BRIGHT = 4


'''
The filtering generated 975 dark images (DI) and 9720 bright ones (BI)
975x9 = 8775 random images from BI will be chosen and each image in DI will be combined
with 9 images from that selection.

Further, each 1944 (=9720/5) of randomly selected images from BI will combine with 4 images from the remaining 7776.

Total examples generated = 7776 + 8775 = 16551, with dark backgrounds making up ~53%
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
brights = bright_files[np.random.permutation(bright_files.shape[0])]
boundary_ind = int(bright_files.shape[0]/(BRIGHTS_PER_BRIGHT+1))
# print(boundary_ind)
brights_b = bright_files[bright_ordering[bright_files.shape[0] - boundary_ind:bright_files.shape[0]]]
brights_r = bright_files[bright_ordering[0:bright_files.shape[0] - boundary_ind]]
brights_r = brights_r[np.random.permutation(brights_r.shape[0])]
brights_for_darks = bright_files[np.random.permutation(dark_files.shape[0]*BRIGHTS_PER_DARK)]

all_files = []
# generating those having dark backgrounds
count = 0
for dark in darks:
    b = reshape_img(dark)
    for i in range(BRIGHTS_PER_DARK):
        r = reshape_img(brights_for_darks[count])
        fname = export_synthetic(b,r,dark,brights_for_darks[count])
        print(fname)
        all_files.append(fname)
        count+=1

# generating the rest
count = 0
for bright in brights_b:
    b = reshape_img(bright)
    for i in range(BRIGHTS_PER_BRIGHT):
        r = reshape_img(brights_r[count])
        fname = export_synthetic(b,r,bright,brights_r[count])
        print(fname)
        all_files.append(fname)
        count+=1

# write onto text file
with open(ALL_SYNTHS_LIST,"w") as list_file:
    for f in all_files:
        list_file.write(f)
        list_file.write('\n')
