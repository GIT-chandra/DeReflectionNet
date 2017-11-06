from keras.models import Model
from keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, Dropout, Concatenate
from keras.layers.core import Reshape, Lambda
from keras.optimizers import Adam
from keras import backend as K
from keras.layers.advanced_activations import  PReLU
from Classes_DeReflectionNet import *
import numpy as np
import keras, cv2

TRAIN_FILES = './train_imgs/*'
VAL_FILES = './val_imgs/*'

TRAIN_IMAGES_LIST = 'train_imgs.txt'
VAL_IMAGES_LIST = 'val_imgs.txt'

WEIGHTS_FILE = 'DeReflectionNet_A.h5'
VGG_WEIGHTS = 'vgg16_weights_tf_dim_ordering_tf_kernels.h5'

class TrainHistory(keras.callbacks.Callback):
    def __init__(self):
        self.train_losses = []
        self.train_accs = []
        self.val_losses = []
        self.val_accs = []

    def on_batch_end(self, batch, logs={}):
        self.train_losses.append(logs.get('loss'))
        self.train_accs.append(logs.get('acc'))

    def on_epoch_end(self, epoch, logs={}):
        self.val_losses.append(logs.get('val_loss'))
        self.val_accs.append(logs.get('val_acc'))
        print('Saving Weights..')
        self.model.save_weights(WEIGHTS_FILE)
        print('Saved.')


class DeReflectionNet:
    def __init__(self,dim_x = 224, dim_y = 224, dim_z = 5, batch_size = 32, num_epochs = 30, shuffle = True, auxiliary = False):
        self.data_params = {'dim_x': dim_x,
                        'dim_y':dim_y,
                        'dim_z':dim_z,
                        'batch_size':batch_size,
                        'shuffle':shuffle,
                        'auxiliary':auxiliary}
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim_z = dim_z
        self.batch_size = batch_size
        self.num_epochs = num_epochs

        self.partition = {}
        self.__load_data()
        self.history = TrainHistory()
        self.training_generator = DataGenerator(**self.data_params).generate(self.partition['train'])
        self.validation_generator = DataGenerator(**self.data_params).generate(self.partition['validation'])
        self.model = self.__get_model()

    def __load_data(self):
        self.partition = {'train':[],'validation':[]}

        with open(TRAIN_IMAGES_LIST,'r') as files_list:
            self.partition['train'] = files_list.read().splitlines()
        with open(VAL_IMAGES_LIST,'r') as files_list:
            self.partition['validation'] = files_list.read().splitlines()

    def __extract_image(self,inp):
        return inp[:,:,:,0:3]

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
        x = Conv2D(64, (3, 3), activation='relu', padding='same', trainable = False ,name='block1_conv1')(img_input)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', trainable = False , name='block1_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

        # Block 2
        x = Conv2D(128, (3, 3), activation='relu', padding='same', trainable = False , name='block2_conv1')(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same', trainable = False , name='block2_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

        # Block 3
        x = Conv2D(256, (3, 3), activation='relu', padding='same', trainable = False , name='block3_conv1')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', trainable = False , name='block3_conv2')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', trainable = False , name='block3_conv3')(x)
        features_A = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

        x2a = self.__get_upconv(256,(4,4),(4,4),'valid','VggA_Upconv',features_A)
        x2a = self.__get_conv(64,(1,1),'valid','VggA_conv',x2a)

        xa = Concatenate(axis=3)([x1a,x2a])
        print(xa.shape)
        xa = self.__get_conv(64,(5,5),'same','ConvA_1',xa)
        xa = self.__get_conv(64,(5,5),'same','ConvA_2',xa)
        xa = self.__get_conv(64,(5,5),'same','ConvA_3',xa)
        xa = self.__get_conv(64,(5,5),'same','ConvA_4',xa)
        outA = self.__get_upconv(3,(2,2),(2,2),'valid','OutA',xa)

        outputs = self.__get_conv(3,(1,1),'valid','Final_conv',outA,False)
        model = Model(input = inputs, output = outputs)
        model.compile(optimizer = Adam(lr = 1e-4, decay = 0.0005), loss = 'mean_squared_error', metrics = ['accuracy'])
        model.summary()
        return model

    def train_model(self):
        print("Loading weights for VGG16 ...")
        self.model.load_weights(VGG_WEIGHTS,by_name = True)
        print("Loaded weights.")
        self.model.fit_generator(generator = self.training_generator,
                                steps_per_epoch = len(self.partition['train'])//self.batch_size,
                                epochs = self.num_epochs,
                                callbacks = [self.history],
                                validation_data = self.validation_generator,
                                validation_steps = len(self.partition['validation'])//self.batch_size,
                                workers = 1)

if __name__ == '__main__':
    myDRnet = DeReflectionNet()
    print("Loading weights for A-net ...")
    myDRnet.model.load_weights('DeReflectionNet_A_30_epochs.h5')
    print("Loaded weights.")
    myDRnet.train_model()

    # ID = 'sample.jpg'
    # cv_img = cv2.imread(ID)
    # img = np.zeros(cv_img.shape)
    # img += cv_img
    # X = np.empty((1,224,224,5))
    # X[0,:,:,0:3] = img
    # X[0,:,:,3] = get_gradient(ID[:-4] + "_b.jpg")
    # X[0,:,:,4] = get_gradient(ID[:-4] + "_r.jpg")
    #
    # res = myDRnet.model.predict(X)
