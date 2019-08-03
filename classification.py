import numpy as np
from keras.datasets import mnist
from util import classify


def estimate_accuracy(num_images):
    _, (x_test, y_test) = mnist.load_data()

    x_test = x_test[:num_images]
    y_test = y_test[:num_images]

    predictions = classify(models_dir='trained/mnist_models', images=x_test)
    return np.mean(y_test == predictions)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_images', type=int, default=1000)
    args = parser.parse_args()

    accuracy = estimate_accuracy(args.num_images)

    print('Classification accuracy on a test set:', accuracy)
