from util import sample, shrink, sample_mse
from keras.models import load_model
from keras.preprocessing.image import array_to_img
import config
import numpy as np

model = load_model('trained/cifar_models/model_0.h5')

n = 10
output_size = 128

for i in range(n):
    #pixels = sample(model, config.target_size, config.num_classes)
    pixels = sample_mse(model, config.target_size, config.num_classes)

    a = np.array(pixels * config.factor, dtype=np.uint8)

    a = a.reshape((config.target_size, config.target_size, 1))
    image = array_to_img(shrink(a, size=output_size))
    image.show()
