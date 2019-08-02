import numpy as np
from skimage import transform
from skimage.util import img_as_ubyte
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import GRU, TimeDistributed, Dense, Activation
import os
from keras.models import load_model


def create_model(one_hot_len):
    model = Sequential()

    model.add(
        GRU(units=256, input_shape=(None, one_hot_len), return_sequences=True)
    )
    model.add(TimeDistributed(Dense(units=one_hot_len)))
    model.add(Activation(activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    return model


def shrink(im, size):
    return img_as_ubyte(transform.resize(im, (size, size)))


def shrink_all(x, size):
    resized = []
    for i in range(len(x)):
        im = shrink(x[i], size)
        resized.append(im)

    return np.array(resized)


def pre_process(images):
    import config

    images = shrink_all(images, config.target_size)
    images = np.array(np.round(images / config.factor), dtype=np.uint8)

    h, w = images[0].shape

    xs = to_categorical(images, num_classes=config.num_classes).reshape(
        -1, h ** 2, config.num_classes
    )

    return xs


def train_model(xs, num_classes, batch_size=32, epochs=50):
    model = create_model(num_classes)

    m = len(xs)

    xs = np.hstack(
        (np.zeros((m, 1, num_classes)), xs[:, 0:, :])
    )

    ys = np.hstack(
        (xs[:, 1:, :], np.zeros((m, 1, num_classes)))
    )

    model.fit(xs, ys, batch_size=batch_size, epochs=epochs)
    return model


def sample(model, image_size, num_classes):
    pixels = np.zeros((image_size ** 2, 1), dtype=np.uint8)
    for i in range(1, image_size ** 2):
        prefix = to_categorical(
            pixels[:i].reshape(1, i), num_classes=num_classes
        ).reshape(1, i, num_classes)

        indices = list(range(num_classes))
        indx = np.random.choice(indices, p=model.predict(prefix)[0][-1])
        yhat = indx
        pixels[i] = yhat

    return pixels


def classify(models_dir, images):
    xs = pre_process(images)
    pixels = np.array(np.argmax(xs, axis=2), dtype=np.uint32)

    m, Tx, n = xs.shape

    xs_extra = np.hstack(
        (np.zeros((m, 1, n)), xs[:, 0:, :])
    )

    image_indices = np.array([[i] * Tx for i in range(m)], dtype=np.uint32)
    sequence_indices = np.zeros((m, Tx), dtype=np.uint32)
    sequence_indices[:] = np.arange(Tx)

    prob_x = np.zeros((10, m))

    for k in range(10):
        path = os.path.join(models_dir, 'model_{}.h5'.format(k))
        model = load_model(path)

        pmf_seqs = model.predict(xs_extra)

        probabilities = pmf_seqs[image_indices, sequence_indices, pixels]

        prob_x[k] = np.prod(probabilities, axis=1)

    print(prob_x.max())
    return np.argmax(prob_x, axis=0)
