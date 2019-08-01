from util import sample, shrink
from keras.models import load_model
from keras.preprocessing.image import array_to_img
import config
import numpy as np
from PIL import ImageDraw, Image


def generate(digit=0, output_size=32, grid_size=5):
    model = load_model('trained/mnist_models/model_{}.h5'.format(digit))

    a = np.zeros((output_size * grid_size, output_size * grid_size),
                 dtype=np.uint8)

    im = Image.fromarray(a, mode='L')

    canvas = ImageDraw.ImageDraw(im)

    for i in range(grid_size):
        for j in range(grid_size):
            pixels = sample(model, config.target_size, config.num_classes)

            a = np.array(pixels * config.factor, dtype=np.uint8)

            a = a.reshape((config.target_size, config.target_size, 1))
            image = array_to_img(shrink(a, size=output_size))
            canvas.bitmap((j * output_size, i * output_size), image, fill=255)

    im.show()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--digit', type=int, default=0)
    args = parser.parse_args()

    generate(digit=args.digit)
