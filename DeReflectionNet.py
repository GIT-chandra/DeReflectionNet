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
