import shutil
import numpy as np

ALL_SYNTHS_LIST = 'synth_imgs.txt'

TRAIN_STOP = 10592
VAL_STOP = TRAIN_STOP + 2648

'''
16551 examples : 10592 train, 2648 val and 3311 test (80/20 splits)
'''

with open(ALL_SYNTHS_LIST,'r') as img_list:
    files = np.array(img_list.read().splitlines())

files = files[np.random.permutation(files.shape[0])]

print("Copying training examples ...")
for i in range(TRAIN_STOP):
    shutil.copy(files[i],'./train_imgs/')
    shutil.copy(files[i][:-4] + "_r.jpg",'./train_imgs/')
    shutil.copy(files[i][:-4] + "_b.jpg",'./train_imgs/')

print("Copying validation examples ...")
for i in range(TRAIN_STOP,VAL_STOP):
    shutil.copy(files[i],'./val_imgs/')
    shutil.copy(files[i][:-4] + "_r.jpg",'./val_imgs/')
    shutil.copy(files[i][:-4] + "_b.jpg",'./val_imgs/')

print("Copying testing examples ...")
for i in range(VAL_STOP,files.shape[0]):
    shutil.copy(files[i],'./test_imgs/')
    shutil.copy(files[i][:-4] + "_r.jpg",'./test_imgs/')
    shutil.copy(files[i][:-4] + "_b.jpg",'./test_imgs/')
