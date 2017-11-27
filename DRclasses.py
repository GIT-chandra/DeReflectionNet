import numpy as np
import cv2
from keras.preprocessing import image
from keras.models import Model

def get_gradient(img_path):
    img = cv2.imread(img_path)
    i = np.zeros(img.shape)
    i += np.array(img)

    i_left = np.zeros(img.shape)
    i_right = np.zeros(img.shape)
    i_up = np.zeros(img.shape)
    i_down = np.zeros(img.shape)

    h = img.shape[0]
    w = img.shape[1]

    i_left[:,0:w-1,:] = i[:,1:w,:]
    i_right[:,1:w,:] = i[:,0:w-1,:]
    i_up[0:h-1,:,:] = i[1:h,:,:]
    i_down[1:h,:,:] = i[0:h-1,:,:]

    i_sum = abs(i-i_left) + abs(i-i_right) + abs(i-i_up) + abs(i-i_down)
    i_sum = np.sum(i_sum,axis=2)
    i_sum /= 4

    return i_sum

class DataGenerator:
    def __init__(self,dim_x = 224, dim_y = 224, dim_z = 4, batch_size = 16, shuffle = True):
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim_z = dim_z
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __get_exploration_order(self, list_IDs):
        indexes = np.arange(len(list_IDs))
        if self.shuffle == True:
            np.random.shuffle(indexes)
        return indexes

    def __data_generation(self,list_IDs_temp):
        X = np.empty((self.batch_size,self.dim_x,self.dim_y,self.dim_z))
        y = np.empty((self.batch_size,self.dim_x,self.dim_y,self.dim_z))

        for i, ID in enumerate(list_IDs_temp):
            cv_img = cv2.imread(ID)
            img = np.zeros(cv_img.shape)
            img += cv_img
            X[i,:,:,0:3] = img/255
            X[i,:,:,3] = get_gradient(ID)/255

            label_fname = ID[:-4] + "_b.jpg"

            cv_back = cv2.imread(label_fname)
            back = np.zeros(cv_back.shape)
            back += cv_back
            y[i,:,:,0:3] = back/255
            y[i,:,:,3] = get_gradient(label_fname)/255

        return X,y

    def generate(self,list_IDs):
        while 1:
            indexes = self.__get_exploration_order(list_IDs)

            imax = int(len(indexes)/self.batch_size)
            for i in range(imax):
                list_IDs_temp = [list_IDs[k] for k in indexes[i*self.batch_size:(i+1)*self.batch_size]]

                X, y = self.__data_generation(list_IDs_temp)
                yield X,y
