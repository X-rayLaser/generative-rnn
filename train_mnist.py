import numpy as np
from keras.datasets import mnist
import config
from util import shrink_all, train_model

mnist_train, (x_test, y_test) = mnist.load_data()
x_train, y_train = mnist_train

x_train = x_train[:10000]
y_train = y_train[:10000]

im_size = config.target_size

x_train = x_train[(y_train == 0)]
x_train = shrink_all(x_train, im_size)
x_train = np.array(np.round(x_train / config.factor), dtype=np.uint8)

model = train_model(x_train, config.num_classes)

model.save('trained/mnist_models/model_0.h5')
