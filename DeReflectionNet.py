from vgg16 import VGG16
from keras.preprocessing import image
from imagenet_utils import preprocess_input
from keras.models import Model
from Classes_DeReflectionNet import *
import numpy as np

class DeReflectionNet:
    def __init__(self,dim_x = 224, dim_y = 224, dim_z = 5, batch_size = 16, shuffle = True, auxiliary = False):
        self.data_params = {'dim_x': dim_x,
                        'dim_y':dim_y,
                        'dim_z':dim_z,
                        'batch_size':batch_size,
                        'shuffle':shuffle,
                        'auxiliary':auxiliary}
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim_z = dim_z

        base_model = VGG16(weights='imagenet')
        self.vgg_A = Model(input=base_model.input, output=base_model.get_layer('block3_pool').output)
        self.vgg_S = Model(input=base_model.input, output=base_model.get_layer('block5_pool').output)
        self.partition = {}
        self.load_data()
        self.training_generator = DataGenerator(**params).generate(self.partition['train'])
        self.validation_generator = DataGenerator(**params).generate(self.partition['validation'])
        self.net = self.get_model()

    def load_data(self):
        return

    def get_model(self):
        inputs = Input(shape = (self.dim_x,self.dim_y,self.dim_z))

        only_rgb = inputs[:,:,0:self.dim_z-2]

        img_arr = image.img_to_array(only_rgb)
        img_arr = np.expand_dims(img_arr, axis=0)
        img_arr = preprocess_input(img_arr)

        features_A = self.vggA.predict(img_arr)
        features_S = self.vggS.predict(img_arr)

        
