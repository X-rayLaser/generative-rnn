import numpy as np
from keras.datasets import mnist
from util import classify

_, (x_test, y_test) = mnist.load_data()

x_test = x_test[:100]
y_test = y_test[:100]

predictions = classify(models_dir='trained/mnist_models', images=x_test)

print('Classification accuracy:', np.mean(y_test == predictions))
