import numpy as np
from keras.datasets import mnist
import config
from util import shrink_all, train_model


def train(digit=0, num_images=100, epochs=50):
    mnist_train, (x_test, y_test) = mnist.load_data()
    x_train, y_train = mnist_train

    x_train = x_train
    y_train = y_train

    im_size = config.target_size

    x_train = x_train[(y_train == digit)]

    x_train = x_train[:num_images]

    x_train = shrink_all(x_train, im_size)
    x_train = np.array(np.round(x_train / config.factor), dtype=np.uint8)

    model = train_model(x_train, config.num_classes, epochs=epochs)

    model.save('trained/mnist_models/model_{}.h5'.format(digit))


if __name__ == '__main__':
    import argparse
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--digit', type=int, default=0)
    parser.add_argument('--num_images', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=50)

    args = parser.parse_args()

    train(digit=args.digit, num_images=args.num_images, epochs=args.epochs)
