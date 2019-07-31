import numpy as np
from keras.datasets import cifar10
import config
from util import shrink_all, train_model

cifar_train, (x_test, y_test) = cifar10.load_data()
x_train, y_train = cifar_train

x_train = x_train[:1000]
y_train = y_train[:1000]
im_size = config.target_size

x_train = x_train[(y_train == 0).ravel()]
x_train = shrink_all(x_train, im_size)

red = x_train[:, :, :, 0]
green = x_train[:, :, :, 1]
blue = x_train[:, :, :, 2]

gray = 0.21 * red + 0.72 * green + 0.07 * blue

x_train = np.array(np.round(gray / config.factor), dtype=np.uint8)

model = train_model(x_train, config.num_classes)

model.save('trained/cifar_models/model_0.h5')
