import shutil
import numpy as np

ALL_SYNTHS_LIST = 'synth_imgs.txt'
TRAIN_IMAGES_LIST = 'train_imgs.txt'
VAL_IMAGES_LIST = 'val_imgs.txt'
TEST_IMAGES_LIST = 'test_imgs.txt'

TRAIN_STOP = 342000
VAL_STOP = TRAIN_STOP + 38000

'''
390000 examples : 342000  train and 38000 val (90/10 splits on 380000), 10000 test
'''

with open(ALL_SYNTHS_LIST,'r') as img_list:
    files = np.array(img_list.read().splitlines())

files = files[np.random.permutation(files.shape[0])]


with open(TRAIN_IMAGES_LIST,"w") as list_file:
    for i in range(TRAIN_STOP):
        shutil.copy(files[i],'./train_imgs/')
        shutil.copy(files[i][:-4] + "_r.jpg",'./train_imgs/')
        shutil.copy(files[i][:-4] + "_b.jpg",'./train_imgs/')
        fname = files[i].split('/')
        fname = fname[len(fname) - 1]
        list_file.write('./train_imgs/' + fname)
        list_file.write('\n')
        print("Copying training examples ..." + str(i) + "/342000")

with open(VAL_IMAGES_LIST,"w") as list_file:
    for i in range(TRAIN_STOP,VAL_STOP):
        shutil.copy(files[i],'./val_imgs/')
        shutil.copy(files[i][:-4] + "_r.jpg",'./val_imgs/')
        shutil.copy(files[i][:-4] + "_b.jpg",'./val_imgs/')
        fname = files[i].split('/')
        fname = fname[len(fname) - 1]
        list_file.write('./val_imgs/' + fname)
        list_file.write('\n')
        print("Copying validation examples ..." + str(i) + "/38000")

with open(TEST_IMAGES_LIST,"w") as list_file:
    for i in range(VAL_STOP,files.shape[0]):
        shutil.copy(files[i],'./test_imgs/')
        shutil.copy(files[i][:-4] + "_r.jpg",'./test_imgs/')
        shutil.copy(files[i][:-4] + "_b.jpg",'./test_imgs/')
        fname = files[i].split('/')
        fname = fname[len(fname) - 1]
        list_file.write('./test_imgs/' + fname)
        list_file.write('\n')
        print("Copying testing examples ..." + str(i) + "/10000")
