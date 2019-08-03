import numpy as np
from keras.datasets import mnist
import config
from util import shrink_all, train_model, pre_process, classify

_, (x_test, y_test) = mnist.load_data()

x_test = x_test[:10000]
y_test = y_test[:10000]

predictions = classify(models_dir='trained/mnist_models', images=x_test)

print('Classification accuracy:', np.mean(y_test == predictions))
