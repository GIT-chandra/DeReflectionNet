from keras.models import Model
from keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, Dropout, Concatenate
from keras.layers.core import Reshape, Lambda
from keras.optimizers import Adam
from keras import backend as K
from keras.layers.advanced_activations import  PReLU
from Classes_DeReflectionNet import *
import numpy as np
import tensorflow as tf
import keras

class DeReflectionNet:
    def __init__(self,dim_x = 224, dim_y = 224, dim_z = 10, batch_size = 16, shuffle = True, auxiliary = False):
        self.data_params = {'dim_x': dim_x,
                        'dim_y':dim_y,
                        'dim_z':dim_z,
                        'batch_size':batch_size,
                        'shuffle':shuffle,
                        'auxiliary':auxiliary}
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim_z = dim_z

        self.partition = {}
        self.__load_data()
        self.training_generator = DataGenerator(**self.data_params).generate(self.partition['train'])
        self.validation_generator = DataGenerator(**self.data_params).generate(self.partition['validation'])
        self.net = self.__get_model()

    def __load_data(self):
        self.partition = {'train':[],'validation':[]}

    def __extract_image(self,inp):
        return inp[:,:,:,0:3]

    def __extract_fA(self,inp):
        data =  inp[:,:,:,5:9]
        return Reshape((28,28,256))(data)

    def  __get_conv(self, num_filters, _size, padding_scheme, layer_name, inp_layer, drop = True):
        conv = Conv2D(num_filters, _size, activation = 'linear', padding = padding_scheme, name = layer_name, kernel_initializer = 'glorot_normal')(inp_layer)
        if drop == False:
            return PReLU()(conv)
        return Dropout(0.5)(PReLU()(conv))

    def  __get_upconv(self, num_filters, _size, _stride, padding_scheme, layer_name, inp_layer, drop = True):
        upconv = Conv2DTranspose(num_filters, _size, strides = _stride, activation = 'linear', padding = padding_scheme, name = layer_name, kernel_initializer = 'glorot_normal')(inp_layer)
        if drop == False:
            return PReLU()(upconv)
        return Dropout(0.5)(PReLU()(upconv))

    def __get_model(self):
        inputs = Input(shape = (self.dim_x,self.dim_y,self.dim_z))

        x1a = self.__get_conv(96,(9,9),'same','PreA_Conv',inputs)
        x1a = MaxPooling2D(pool_size=(2,2))(x1a)

        img_input = Lambda(self.__extract_image, output_shape = (self.dim_x,self.dim_y,3))(inputs)

        #  Here goes VGG16
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

        # Block 2
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

        # Block 3
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
        features_A = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

        # Block 4
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(features_A)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

        # Block 5
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
        features_S = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

        x2a = self.__get_upconv(256,(4,4),(4,4),'valid','VggA_Upconv',features_A)
        x2a = self.__get_conv(64,(1,1),'valid','VggA_conv',x2a)

        xa = Concatenate(axis=3)([x1a,x2a])
        xa = self.__get_conv(64,(5,5),'same','ConvA_1',xa)
        xa = self.__get_conv(64,(5,5),'same','ConvA_2',xa)
        xa = self.__get_conv(64,(5,5),'same','ConvA_3',xa)
        xa = self.__get_conv(64,(5,5),'same','ConvA_4',xa)
        outA = self.__get_upconv(3,(2,2),(2,2),'valid','OutA',x)

        x1s = self.__get_conv(96,(9,9),'same','PreS_Conv',inputs)
        x1s = MaxPooling2D(pool_size=(2,2))(x1s)

        x2s = self.__get_upconv(256,(4,4),(4,4),'valid','VggS_Upconv1',features_S)
        x2s = self.__get_upconv(256,(4,4),(4,4),'valid','VggS_Upconv2',x2s)
        x2s = self.__get_conv(64,(1,1),'valid','VggS_conv',x2s)
        xs = Concatenate(axis=3)([x1s,x2s])
        xa = self.__get_conv(64,(5,5),'same','ConvS_1',xs)
        xa = self.__get_conv(64,(5,5),'same','ConvS_2',xs)
        xa = self.__get_conv(64,(5,5),'same','ConvS_3',xs)
        xa = self.__get_conv(64,(5,5),'same','ConvS_4',xs)
        outS = self.__get_upconv(3,(2,2),(2,2),'valid','OutS',x)

        outputs = Concatenate(axis=3)([outA,outS])
        outputs = self.__get_conv(3,(1,1),'valid','Final_conv',outputs,False)
        model = Model(input = inputs, output = outputs)
        model.summary()
        return model


if __name__ == '__main__':
    myDRnet = DeReflectionNet()
