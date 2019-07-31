import numpy as np
from skimage import transform
from skimage.util import img_as_ubyte
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import GRU, TimeDistributed, Dense, Activation


def shrink(im, size):
    return img_as_ubyte(transform.resize(im, (size, size)))


def shrink_all(x, size):
    resized = []
    for i in range(len(x)):
        im = shrink(x[i], size)
        resized.append(im)

    return np.array(resized)


def train_model(images, num_classes, batch_size=32, epochs=50):
    model = Sequential()

    model.add(GRU(units=256, input_shape=(None, num_classes), return_sequences=True))
    model.add(TimeDistributed(Dense(units=num_classes)))
    model.add(Activation(activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    h, w = images[0].shape

    xs = to_categorical(images, num_classes=num_classes).reshape(
        -1, h ** 2, num_classes
    )

    m = len(xs)

    ys = np.hstack(
        (xs[:, 1:, :], np.zeros((m, 1, num_classes)))
    )

    model.fit(xs, ys, batch_size=batch_size, epochs=epochs)
    return model


def visualize_predictions(images):
    from PIL import ImageDraw, Image


def sample(model, image_size, num_classes):
    pixels = np.zeros((image_size ** 2, 1), dtype=np.uint8)
    for i in range(1, image_size ** 2):
        prefix = to_categorical(pixels[:i].reshape(1, i), num_classes=num_classes).reshape(1, i, num_classes)

        indices = list(range(num_classes))
        indx = np.random.choice(indices, p=model.predict(prefix)[0][-1])
        yhat = indx
        #yhat = model.predict_classes(prefix)[0][-1]
        pixels[i] = yhat

    return pixels
