from keras.datasets import cifar10
from PIL import Image
import numpy as np


def to_pil_image(np_img):
    # np_img dimension: (height, width, channel)
    return Image.fromarray((np_img * 255).astype(np.uint8))

(x_train, _), (x_test, _) = cifar10.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

with open('train.txt', 'w') as f:
    for i, image in enumerate(x_train):
        img = to_pil_image(image)
        filename = 'images/train/%d.jpg' % i
        print(filename)
        f.write(filename + "\n")
        img.save(filename)

with open('test.txt', 'w') as f:
    for i, image in enumerate(x_test):
        img = to_pil_image(image)
        filename = 'images/test/%d.jpg' % i
        print(filename)
        f.write(filename + "\n")
        img.save(filename)
