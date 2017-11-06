import numpy as np
import cv2
from vgg16 import VGG16
from keras.preprocessing import image
from imagenet_utils import preprocess_input
from keras.models import Model

from tools import threadsafe_generator

# base_model = VGG16(weights='imagenet')
# vgg_A = Model(input=base_model.input, output=base_model.get_layer('block3_pool').output)
# vgg_S = Model(input=base_model.input, output=base_model.get_layer('block5_pool').output)
#
# def get_features_A(img_path,A_net = True):
#     img = image.load_img(img_path, target_size=(224, 224))
#     x = image.img_to_array(img)
#     x = np.expand_dims(x, axis=0)
#     x = preprocess_input(x)
#     if A_net == False:
#         return vgg_S.predict(x)
#     return vgg_S.predict(x)

# def get_features_A(img_path):
#     img = image.load_img(img_path, target_size=(224, 224))
#     x = image.img_to_array(img)
#     x = np.expand_dims(x, axis=0)
#     x = preprocess_input(x)
#     return vgg_A.predict(x)

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
    def __init__(self,dim_x = 224, dim_y = 224, dim_z = 5, batch_size = 16, shuffle = True, auxiliary = False):

        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim_z = dim_z
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.auxiliary = auxiliary

    def __get_exploration_order(self, list_IDs):
        indexes = np.arange(len(list_IDs))
        if self.shuffle == True:
            np.random.shuffle(indexes)
        return indexes

    def __data_generation(self,list_IDs_temp):
        X = np.empty((self.batch_size,self.dim_x,self.dim_y,self.dim_z))
        y = np.empty((self.batch_size,self.dim_x,self.dim_y,3))

        for i, ID in enumerate(list_IDs_temp):
            cv_img = cv2.imread(ID)
            img = np.zeros(cv_img.shape)
            img += cv_img
            X[i,:,:,0:3] = img
            X[i,:,:,3] = get_gradient(ID[:-4] + "_b.jpg")
            X[i,:,:,4] = get_gradient(ID[:-4] + "_r.jpg")
            # f_A = get_features_A(ID + ".jpg")
            # f_S = get_features_S(ID + ".jpg")
            # X[i,:,:,5:9,0] = f_A.reshape((self.dim_x,self.dim_y,4))
            # temp = np.zeros((self.dim_x,self.dim_y))
            # temp[:,0:112] = f_S.reshape((224,112))
            # X[i,:,:,9,0] = temp

            label_fname = ID[:-4] + "_b.jpg"
            if self.auxiliary == True:
                label_fname = ID[:-4] + "_r.jpg"

            cv_back = cv2.imread(label_fname)
            back = np.zeros(cv_back.shape)
            back += cv_back
            y[i,:,:,:] = back

        return X,y

    # @threadsafe_generator
    def generate(self,list_IDs):
        while 1:
            indexes = self.__get_exploration_order(list_IDs)

            imax = int(len(indexes)/self.batch_size)
            for i in range(imax):
                list_IDs_temp = [list_IDs[k] for k in indexes[i*self.batch_size:(i+1)*self.batch_size]]

                X, y = self.__data_generation(list_IDs_temp)
                yield X,y

if __name__ == '__main__':
    f = get_features_A('sample.jpg',False)
    print(type(f))
    print(f.shape)
