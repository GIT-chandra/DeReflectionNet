from keras.models import Model
from keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, Dropout, Concatenate, Activation
from keras.layers.core import Reshape, Lambda
from keras.optimizers import Adam
from keras import backend as K
from keras.layers.normalization import BatchNormalization
from DRclasses import *
import numpy as np
import keras, cv2, glob

TRAIN_FILES = './train_imgs/*'
VAL_FILES = './val_imgs/*'

TRAIN_IMAGES_LIST = 'train_imgs.txt'
VAL_IMAGES_LIST = 'val_imgs.txt'

WEIGHTS_FILE = 'DRNet.h5'
VGG_WEIGHTS = 'vgg16_weights_tf_dim_ordering_tf_kernels.h5'

START_EPOCH = 1
VGG_TRAINABLE = True

class TrainHistory(keras.callbacks.Callback):
    def __init__(self):
        self.count = START_EPOCH
        self.train_losses = []
        self.val_losses = []

    def on_batch_end(self, batch, logs={}):
        self.train_losses.append(logs.get('loss'))

    def on_epoch_end(self, epoch, logs={}):
        self.val_losses.append(logs.get('val_loss'))
        print('Saving Weights..')
        self.model.save_weights(WEIGHTS_FILE[:-3]+str(self.count) + ".h5")
        print('Saved.')
        self.count += 1
        self.save_to_files()

    def save_to_files(self):
        np.save('DRNet_unfrozen_train_losses.npy',self.train_losses)
        np.save('DRNet_unfrozen_val_losses.npy',self.val_losses)
        np.save('LastLR.npy',K.eval(self.model.optimizer.lr))

class DRNet:
    def __init__(self,dim_x = 224, dim_y = 224, dim_z = 3, batch_size = 18, num_epochs = 40, shuffle = True):
        self.data_params = {'dim_x': dim_x,
                        'dim_y':dim_y,
                        'dim_z':dim_z,
                        'batch_size':batch_size,
                        'shuffle':shuffle}
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

    # def __extract_grads(self,inp):
    #     zero_column = K.zeros(shape=(1,224,1,3))
    #     zero_row = K.zeros(shape=(1,1,224,3))
    #     Left = Reshape((224,223,3))(inp[:,:,1:self.dim_x,:])
    #
    #     i_left = Concatenate(axis=2)([Left,zero_column])
    #     i_right = Concatenate(axis=2)([zero_column,inp[:,0:self.dim_x-1,:]])
    #     i_up = Concatenate(axis=1)([inp[1:self.dim_y,:,:],zero_row])
    #     i_down = Concatenate(axis=1)([zero_row,inp[0:self.dim_y-1,:,:]])
    #
    #     i_sum = K.abs(i-i_left) + K.abs(i-i_right) + K.abs(i-i_up) + K.abs(i-i_down)
    #     i_sum = K.sum(i_sum,axis=3)
    #     i_sum /= 4
    #     return Reshape((self.dim_x,self.dim_y,1))(i_sum)

    def __get_conv(self, num_filters, _size, padding_scheme, layer_name, inp_layer, drop = True):
        return Conv2D(num_filters, _size, activation = 'linear', padding = padding_scheme, name = layer_name, kernel_initializer = 'glorot_normal')(inp_layer)

    def __get_BN(self, layer_name, inp_layer):
        bn = BatchNormalization(axis=-1, momentum=0.9, epsilon=0.001, name = layer_name)(inp_layer)
        rl = Activation('relu')(bn)
        return Dropout(0.5)(rl)

    def __get_upconv(self, num_filters, _size, _stride, padding_scheme, layer_name, inp_layer, drop = True):
        return Conv2DTranspose(num_filters, _size, strides = _stride, activation = 'linear', padding = padding_scheme, name = layer_name, kernel_initializer = 'glorot_normal')(inp_layer)

    def __get_model(self):
        inputs = Input(shape = (self.dim_x,self.dim_y,self.dim_z))

        #  Here goes VGG16
        x = Conv2D(64, (3, 3), activation='relu', padding='same' , trainable = VGG_TRAINABLE, name='block1_conv1')(inputs)
        x_1 = Conv2D(64, (3, 3), activation='relu', padding='same' , name='block1_conv2')(x)
        features_1 = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x_1)

        # Block 2
        x = Conv2D(128, (3, 3), activation='relu', padding='same' , trainable = VGG_TRAINABLE, name='block2_conv1')(features_1)
        x_2 = Conv2D(128, (3, 3), activation='relu', padding='same' , name='block2_conv2')(x)
        features_2 = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x_2)

        # Block 3
        x = Conv2D(256, (3, 3), activation='relu', padding='same' , trainable = VGG_TRAINABLE, name='block3_conv1')(features_2)
        x = Conv2D(256, (3, 3), activation='relu', padding='same' , trainable = VGG_TRAINABLE, name='block3_conv2')(x)
        x_3 = Conv2D(256, (3, 3), activation='relu', padding='same' , trainable = VGG_TRAINABLE, name='block3_conv3')(x)
        features_3 = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x_3)

        # Block 4
        x = Conv2D(512, (3, 3), activation='relu', padding='same' , trainable = VGG_TRAINABLE, name='block4_conv1')(features_3)
        x = Conv2D(512, (3, 3), activation='relu', padding='same' , trainable = VGG_TRAINABLE, name='block4_conv2')(x)
        x_4 = Conv2D(512, (3, 3), activation='relu', padding='same' , trainable = VGG_TRAINABLE, name='block4_conv3')(x)
        features_4 = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x_4)

        # Block 5
        x = Conv2D(512, (3, 3), activation='relu', padding='same' , trainable = VGG_TRAINABLE, name='block5_conv1')(features_4)
        x = Conv2D(512, (3, 3), activation='relu', padding='same' , trainable = VGG_TRAINABLE, name='block5_conv2')(x)
        x_5 = Conv2D(512, (3, 3), activation='relu', padding='same' , trainable = VGG_TRAINABLE, name='block5_conv3')(x)
        features_5 = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x_5)
        # --------------------------

        # start restoration to original spatial size
        x7 = self.__get_conv(512,(3,3),'same','conv_l7', features_5)
        x = self.__get_BN('DR_upconv_BN1a',x7)
        x = self.__get_conv(512,(3,3),'same','DR_conv1a',x)
        x = self.__get_BN('DR_upconv_BN1b',x)
        x = self.__get_upconv(512,(2,2),(2,2),'valid','DR_upconv1b',x)
        x = self.__get_BN('DR_upconv_BN1',x)


        x14 = self.__get_conv(512,(3,3),'same','conv_l14', Concatenate(axis=3)([x,x_5]))
        x = self.__get_BN('DR_upconv_BN2a',x14)
        x = self.__get_conv(512,(3,3),'same','DR_conv2a', x)
        x = self.__get_BN('DR_upconv_BN2b',x)
        x = self.__get_upconv(512,(2,2),(2,2),'valid','DR_upconv2b',x)
        x = self.__get_BN('DR_upconv_BN2',x)


        x28 = self.__get_conv(512,(3,3),'same','conv_l28', Concatenate(axis=3)([x,x_4]))
        x = self.__get_BN('DR_upconv_BN3a',x28)
        x = self.__get_conv(512,(3,3),'same','DR_conv3a', x)
        x = self.__get_BN('DR_upconv_BN3b',x)
        x = self.__get_upconv(256,(2,2),(2,2),'valid','DR_upconv3b',x)
        x = self.__get_BN('DR_upconv_BN3',x)


        x56 = self.__get_conv(256,(3,3),'same','conv_l56', Concatenate(axis=3)([x,x_3]))
        x = self.__get_BN('DR_upconv_BN4a',x56)
        x = self.__get_conv(256,(3,3),'same','DR_conv4a', x)
        x = self.__get_BN('DR_upconv_BN4b',x56)
        x = self.__get_upconv(128,(2,2),(2,2),'valid','DR_upconv4b',x)
        x = self.__get_BN('DR_upconv_BN4',x)


        x112 = self.__get_conv(128,(3,3),'same','conv_l112', Concatenate(axis=3)([x,x_2]))
        x = self.__get_BN('DR_upconv_BN5a',x112)
        x = self.__get_upconv(64,(2,2),(2,2),'valid','DR_upconv5b',x)
        x = self.__get_BN('DR_upconv_BN5',x)

        # restored to original spatial size

        x = self.__get_conv(64,(3,3),'same','DR_conv_pre_final1',Concatenate(axis=3)([x,x_1]))
        x = self.__get_BN('DR_final_BN1',x)

        x = self.__get_conv(64,(3,3),'same','DR_conv_pre_final2',x)
        x = self.__get_BN('DR_final_BN2',x)

        outputs = Conv2D(3, (1,1), activation = 'linear', padding = 'same', name = 'DR_conv_final')(x)
        model = Model(input = inputs, output = outputs)
        model.compile(optimizer = Adam(lr = 1e-4, decay = 0.0001), loss = 'mean_squared_error')
        model.summary()
        return model

    def train_model(self):

        if START_EPOCH == 1:
            print("Loading weights for VGG16 ...")
            self.model.load_weights(VGG_WEIGHTS,by_name = True)
            print("Loaded weights.")

        else:
            self.model.load_weights('DRNet'+str(START_EPOCH-1) + '.h5')
            print("Loaded weights.")
            self.model.optimizer.lr.assign(np.load('LastLR.npy'))
            print("Loaded learning rate ...")

        self.model.fit_generator(generator = self.training_generator,
                                steps_per_epoch = len(self.partition['train'])//self.batch_size,
                                epochs = self.num_epochs,
                                callbacks = [self.history],
                                validation_data = self.validation_generator,
                                validation_steps = len(self.partition['validation'])//self.batch_size,
                                workers = 1)
myDRnet_global = DRNet()
# if __name__ == '__main__':
#     myDRnet_global.train_model()

def evt(ID):
    cv_img = cv2.imread(ID)
    img = np.zeros(cv_img.shape)
    img += cv_img

    X = np.empty((1,224,224,3))
    X[0,:,:,0:3] = img/255
    # X[0,:,:,3] = get_gradient(ID[:-4] + "_b.jpg")/255

    res = myDRnet_global.model.predict(X)
    res_img = res.reshape((224,224,3))
    res_img = res_img * 255

    img_back = np.zeros((224,224,3))
    img_back += cv2.imread(ID[:-4] + "_b.jpg")

    r_r = res_img[:,:,0]
    r_g = res_img[:,:,1]
    r_b = res_img[:,:,2]

    i_r = img[:,:,0]
    i_g = img[:,:,1]
    i_b = img[:,:,2]

    r = r_r*np.sum((r_r*i_r).flatten())/np.sum((r_r*r_r).flatten())
    g = r_g*np.sum((r_g*i_g).flatten())/np.sum((r_g*r_g).flatten())
    b = r_b*np.sum((r_b*i_b).flatten())/np.sum((r_b*r_b).flatten())

    c = np.empty((224,224,3))
    c[:,:,0]=r
    c[:,:,1]=g
    c[:,:,2]=b

    cv2.imwrite(ID[:-4] + "_gt.jpg",img - img_back)
    cv2.imwrite(ID[:-4] + "_pred_cc.jpg",c)
    cv2.imwrite(ID[:-4] + "_pred.jpg",res_img)

    cv2.imwrite(ID[:-4] + "_b_pred.jpg",img - res_img)


    # cv2.imwrite(ID[:-4] + "_b_pred_cc.jpg",img - c)
    # return res

print("Loading weights ...")
myDRnet_global.model.load_weights('DRNet31.h5')
print("Loaded weights.")
#
# # for f in glob.glob("./Results/Real_Data/*.jpg"):
# for f in glob.glob("./Results/Test_Data/????_??????_????_??????.jpg"):
#     print(f)
#     evt(f)
