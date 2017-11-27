from keras.models import Model
from keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, Dropout, Concatenate, Activation, Add, ZeroPadding2D
from keras.layers.core import Reshape, Lambda
from keras.optimizers import Adam
from keras import backend as K
from keras.layers.advanced_activations import  PReLU
from keras.layers.normalization import BatchNormalization
from DRclasses import *
import numpy as np
import keras, cv2, glob

TRAIN_FILES = './train_imgs/*'
VAL_FILES = './val_imgs/*'

TRAIN_IMAGES_LIST = 'train_imgs.txt'
VAL_IMAGES_LIST = 'val_imgs.txt'

WEIGHTS_FILE = 'DRNet.h5'
ECNN_WEIGHTS = 'ECNN.h5'
VGG_WEIGHTS = 'vgg16_weights_tf_dim_ordering_tf_kernels.h5'

START_EPOCH = 1
VGG_TRAINABLE = True

#creating a residual block
def residual_block(x_input,l_name):

    conv1 = Conv2D(64, kernel_size=(3,3), strides=(1, 1), padding='same',name ='residual' + l_name + '_conv1')(x_input)
    bn1 = BatchNormalization(name ='residual' + l_name + '_BN1')(conv1)
    act1 = Activation('relu')(bn1)

    conv2 = Conv2D(64, kernel_size=(3,3), strides=(1, 1), padding='same',name ='residual' + l_name + '_conv2')(act1)
    bn2 = BatchNormalization(name ='residual' + l_name + '_BN2')(conv2)

    return Add()([bn2, Activation('relu')(x_input), x_input])

class TrainHistory(keras.callbacks.Callback):
    def __init__(self):
        self.count = START_EPOCH
        self.train_losses = []
        self.val_losses = []

    def on_batch_end(self, batch, logs={}):
        self.train_losses.append(logs.get('loss'))

    def on_epoch_end(self, epoch, logs={}):
        self.val_losses.append(logs.get('val_loss'))
        self.save_to_files()
        self.count += 1

    def __evt(self,ID):
        cv_img = cv2.imread(ID)
        img = np.zeros(cv_img.shape)
        img += cv_img

        X = np.empty((1,224,224,4))
        X[0,:,:,0:3] = img/255
        X[0,:,:,3] = get_gradient(ID)/255

        res = self.model.predict(X)
        res_img = res[:,:,:,0:3].reshape((224,224,3))
        res_img = res_img * 255

        res_grad = res[:,:,:,3].reshape((224,224,1))
        res_grad = res_grad * 255

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

        cv2.imwrite("./misc/results_real" + ID[16:-4] + "_apredGrad_" + str(self.count) + "_.jpg",res_grad)
        cv2.imwrite("./misc/results_real" + ID[16:-4] + "_apred_" + str(self.count) + "_.jpg",res_img)
        cv2.imwrite("./misc/results_real" + ID[16:-4] + "_apred_cc_" + str(self.count) + "_.jpg",c)

    def save_to_files(self):
        # for imgfile in glob.glob("./misc/CEILNet_Imgs/????_??????.jpg"):
        #     print(imgfile)
        #     temp = self.__evt(imgfile)
        for imgfile in glob.glob("./misc/test_real/*.jpg"):
            print(imgfile)
            temp = self.__evt(imgfile)

        np.save(WEIGHTS_FILE[:-3] +'_train_losses.npy',self.train_losses)
        np.save(WEIGHTS_FILE[:-3] +'_val_losses.npy',self.val_losses)
        np.save(WEIGHTS_FILE[:-3] +'_lastLR.npy',K.eval(self.model.optimizer.lr))

class DRNet:
    def __init__(self,dim_x = 224, dim_y = 224, dim_z = 4, batch_size = 8, num_epochs = 20, shuffle = True):
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

    def __extract_image(self,inp):
        return inp[:,:,:,0:3]

    def __extract_grads(self,inp):
        return Reshape((self.dim_x,self.dim_y,1))(inp[:,:,:,3])

    def __get_conv(self, num_filters, _size, padding_scheme, layer_name, inp_layer, drop = True):
        return Conv2D(num_filters, _size, activation = 'linear', padding = padding_scheme, name = layer_name, kernel_initializer = 'glorot_normal')(inp_layer)

    def __get_BN(self, layer_name, inp_layer):
        bn = BatchNormalization(axis=-1, momentum=0.9, epsilon=0.001, name = layer_name)(inp_layer)
        rl = Activation('relu')(bn)
        # return rl
        return Dropout(0.5)(rl)

    def __get_upconv(self, num_filters, _size, _stride, padding_scheme, layer_name, inp_layer, drop = True):
        return Conv2DTranspose(num_filters, _size, strides = _stride, activation = 'linear', padding = padding_scheme, name = layer_name, kernel_initializer = 'glorot_normal')(inp_layer)

    def __get_model(self):
        x_input = Input(shape=(self.dim_x,self.dim_y,self.dim_z))
        img_input = Lambda(self.__extract_image, output_shape = (self.dim_x,self.dim_y,3))(x_input)

        # ECNN
        conv1 = Conv2D(64, kernel_size=(3,3), strides=(1, 1), padding='same', name = 'ECNN_conv1')(x_input)
        bn1 = BatchNormalization(name = 'ECNN_BN1')(conv1)
        act1 = Activation('relu')(bn1)

        conv2 = Conv2D(64, kernel_size=(3,3), strides=(1, 1), padding='same', name = 'ECNN_conv2')(act1)
        bn2 = BatchNormalization(name = 'ECNN_BN2')(conv2)
        act2 = Activation('relu')(bn2)

        pad3 = ZeroPadding2D(padding=(1,1))(act2)
        conv3 = Conv2D(64, kernel_size=(3,3), strides=(2, 2), padding='valid', name = 'ECNN_conv3')(pad3)
        bn3 = BatchNormalization(name = 'ECNN_BN3')(conv3)
        act3 = Activation('relu')(bn3)

        activation = act3
        # attaching 13 residual blocks
        for i in range(13):
            concat = residual_block(activation,str(i))
            activation = concat

        conv_1 = Conv2DTranspose(64, kernel_size=(2,2), strides=(2,2), padding='valid', name = 'ECNN_deconv1')(activation)
        bn_1 = BatchNormalization(name = 'ECNN_BN4')(conv_1)
        act_1 = Activation('relu')(bn_1)

        conv_2 = Conv2D(64, kernel_size=(3,3), strides=(1, 1), padding='same',name = 'ECNN_conv4')(act_1)
        bn_2 = BatchNormalization(name = 'ECNN_BN5')(conv_2)
        act_2 = Activation('relu')(bn_2)

        grad_input = Conv2D(1, kernel_size=(1,1), strides=(1, 1), padding='same',name = 'ECNN_conv5')(act_2)
        #----------------------------------------------------------


        # feature extraction from the gradients
        x = self.__get_conv(32,(3,3),'same','grad_conv1',grad_input)
        x = self.__get_BN('grad_BN1',x)
        grad_features_1 = MaxPooling2D((2, 2), strides=(2, 2))(x)

        x = self.__get_conv(64,(3,3),'same','grad_conv2',grad_features_1)
        x = self.__get_BN('grad_BN2',x)
        grad_features_2 = MaxPooling2D((2, 2), strides=(2, 2))(x)

        x = self.__get_conv(128,(3,3),'same','grad_conv3',grad_features_2)
        x = self.__get_BN('grad_BN3',x)
        grad_features_3 = MaxPooling2D((2, 2), strides=(2, 2))(x)

        x = self.__get_conv(256,(3,3),'same','grad_conv4',grad_features_3)
        x = self.__get_BN('grad_BN4',x)
        grad_features_4 = MaxPooling2D((2, 2), strides=(2, 2))(x)

        x = self.__get_conv(256,(3,3),'same','grad_conv5',grad_features_4)
        x = self.__get_BN('grad_BN5',x)
        grad_features_5 = MaxPooling2D((2, 2), strides=(2, 2))(x)
        # ---------------------------------------------


        #  Here goes VGG16
        x = Conv2D(64, (3, 3), activation='relu', padding='same' , trainable = VGG_TRAINABLE, name='block1_conv1')(img_input)
        x = self.__get_BN('grad_vgg1a',x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same' , name='block1_conv2')(x)
        x = self.__get_BN('grad_vgg1b',x)
        features_1 = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

        # Block 2
        x = Conv2D(128, (3, 3), activation='relu', padding='same' , trainable = VGG_TRAINABLE, name='block2_conv1')(features_1)
        x = self.__get_BN('grad_vgg2a',x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same' , name='block2_conv2')(x)
        x = self.__get_BN('grad_vgg2b',x)
        features_2 = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

        # Block 3
        x = Conv2D(256, (3, 3), activation='relu', padding='same' , trainable = VGG_TRAINABLE, name='block3_conv1')(features_2)
        x = self.__get_BN('grad_vgg3a',x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same' , trainable = VGG_TRAINABLE, name='block3_conv2')(x)
        x = self.__get_BN('grad_vgg3b',x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same' , trainable = VGG_TRAINABLE, name='block3_conv3')(x)
        x = self.__get_BN('grad_vgg3c',x)
        features_3 = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

        # Block 4
        x = Conv2D(512, (3, 3), activation='relu', padding='same' , trainable = VGG_TRAINABLE, name='block4_conv1')(features_3)
        x = self.__get_BN('grad_vgg4a',x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same' , trainable = VGG_TRAINABLE, name='block4_conv2')(x)
        x = self.__get_BN('grad_vgg4b',x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same' , trainable = VGG_TRAINABLE, name='block4_conv3')(x)
        x = self.__get_BN('grad_vgg4c',x)
        features_4 = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

        # Block 5
        x = Conv2D(512, (3, 3), activation='relu', padding='same' , trainable = VGG_TRAINABLE, name='block5_conv1')(features_4)
        x = self.__get_BN('grad_vgg5a',x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same' , trainable = VGG_TRAINABLE, name='block5_conv2')(x)
        x = self.__get_BN('grad_vgg5b',x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same' , trainable = VGG_TRAINABLE, name='block5_conv3')(x)
        x = self.__get_BN('grad_vgg5c',x)
        features_5 = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
        # --------------------------

        # start restoration to original spatial size
        x7 = self.__get_conv(512,(3,3),'same','conv_l7', Concatenate(axis=3)([features_5,grad_features_5]))
        x = self.__get_BN('DR_upconv_BN1a',x7)
        x = self.__get_conv(512,(3,3),'same','DR_conv1a',x7)
        x = self.__get_BN('DR_upconv_BN1b',x)
        x = self.__get_upconv(256,(2,2),(2,2),'valid','DR_upconv1b',x)


        x14 = self.__get_conv(512,(3,3),'same','conv_l14', Concatenate(axis=3)([x,features_4,grad_features_4]))
        x = self.__get_BN('DR_upconv_BN2a',x14)
        x = self.__get_conv(512,(3,3),'same','DR_conv2a', x14)
        x = self.__get_BN('DR_upconv_BN2b',x)
        x = self.__get_upconv(256,(2,2),(2,2),'valid','DR_upconv2b',x)


        x28 = self.__get_conv(256,(3,3),'same','conv_l28', Concatenate(axis=3)([x,features_3,grad_features_3]))
        x = self.__get_BN('DR_upconv_BN3a',x28)
        x = self.__get_conv(256,(3,3),'same','DR_conv3a', x28)
        x = self.__get_BN('DR_upconv_BN3b',x)
        x = self.__get_upconv(128,(2,2),(2,2),'valid','DR_upconv3b',x)


        x56 = self.__get_conv(128,(3,3),'same','conv_l56', Concatenate(axis=3)([x,features_2,grad_features_2]))
        x = self.__get_BN('DR_upconv_BN4a',x56)
        x = self.__get_conv(128,(3,3),'same','DR_conv4a', x56)
        x = self.__get_BN('DR_upconv_BN4b',x)
        x = self.__get_upconv(64,(2,2),(2,2),'valid','DR_upconv4b',x)


        x112 = self.__get_conv(64,(3,3),'same','conv_l112', Concatenate(axis=3)([x,features_1,grad_features_1]))
        x = self.__get_BN('DR_upconv_BN5a',x112)
        x = self.__get_conv(64,(3,3),'same','DR_conv5a', x112)
        x = self.__get_BN('DR_upconv_BN5b',x)
        x = self.__get_upconv(64,(2,2),(2,2),'valid','DR_upconv5b',x)

        # restored to original spatial size

        x = self.__get_conv(64,(3,3),'same','DR_conv_pre_final1',x)
        x = self.__get_BN('DR_final_BN1',x)

        x = self.__get_conv(64,(3,3),'same','DR_conv_pre_final2',x)
        x = self.__get_BN('DR_final_BN2',x)

        outputs = Conv2D(3, (1,1), activation = 'linear', padding = 'same', name = 'DR_conv_final')(x)
        model = Model(input = x_input, output = Concatenate(axis=3)([outputs,grad_input]))
        model.compile(optimizer = Adam(lr = 1e-4, decay = 0.0001), loss = 'mean_squared_error')
        model.summary()
        return model

    def train_model(self):

        print("Loading weights for VGG16 ...")
        self.model.load_weights(VGG_WEIGHTS,by_name = True)
        print("Loaded weights.")

        # print("Loading weights for ECNN...")
        # self.model.load_weights(ECNN_WEIGHTS,by_name = True)
        # print("Loaded weights.")

        self.model.fit_generator(generator = self.training_generator,
                                steps_per_epoch = len(self.partition['train'])//self.batch_size,
                                epochs = self.num_epochs,
                                callbacks = [self.history],
                                validation_data = self.validation_generator,
                                validation_steps = len(self.partition['validation'])//self.batch_size,
                                workers = 1)

if __name__ == '__main__':
    myDRnet = DRNet()
    myDRnet.train_model()

# myDRnet_global = DRNet()
# print("Loading weights ...")
# myDRnet_global.model.load_weights('DRNet30.h5')
# print("Loaded weights.")
def evt(ID):
    cv_img = cv2.imread(ID)
    img = np.zeros(cv_img.shape)
    img += cv_img

    X = np.empty((1,224,224,4))
    X[0,:,:,0:3] = img/255
    X[0,:,:,3] = get_gradient(ID[:-4] + "_b.jpg")/255

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

    cv2.imwrite(ID[:-4] + "_pred_ccrected.jpg",c)
    cv2.imwrite(ID[:-4] + "_pred.jpg",res_img)
    return res

# import glob
# for imgfile in glob.glob("./val_imgs_old/????_??????_????_??????.jpg"):
#     print(imgfile)
#     temp = evt(imgfile)
