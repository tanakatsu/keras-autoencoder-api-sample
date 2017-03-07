# encoding: utf-8
from PIL import Image
import numpy as np

IMAGE_SIZE = 32


def to_np_data_array(img):
    if img.mode != "RGB":
        img = img.convert("RGB")
    # return (np.asarray(img).astype(np.float32) / 255).transpose(2, 0, 1)  # chainer
    return (np.asarray(img).astype(np.float32) / 255)  # tensorflow


class DataSets(object):

    def __init__(self, train_data='train.txt', test_data='test.txt'):
        self.train_data = train_data
        self.test_data = test_data

    def load_data(self):
        train_images = []
        test_images = []

        with open(self.train_data, "r") as f:
            for line in f:
                line = line.strip()
                img = Image.open(line)
                img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
                img_data = to_np_data_array(img)
                train_images.append(img_data)

        with open(self.test_data, "r") as f:
            for line in f:
                line = line.strip()
                img = Image.open(line)
                img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
                img_data = to_np_data_array(img)
                test_images.append(img_data)

        return np.asarray(train_images), np.asarray(test_images)

if __name__ == '__main__':
    datasets = DataSets()
    x_train, x_test = datasets.load_data()
    print(x_train.shape, x_test.shape)
