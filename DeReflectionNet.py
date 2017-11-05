from vgg16 import VGG16
from keras.preprocessing import image
from imagenet_utils import preprocess_input
from keras.models import Model
import numpy as np

base_model = VGG16(weights='imagenet')
vgg_A = Model(input=base_model.input, output=base_model.get_layer('block3_pool').output)
vgg_S = Model(input=base_model.input, output=base_model.get_layer('block5_pool').output)

img_path = 'cat.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

vggA_features = vgg_A.predict(x)
vggS_features = vgg_S.predict(x)
