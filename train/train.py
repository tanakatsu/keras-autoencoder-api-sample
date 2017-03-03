from keras.layers import Input, Convolution2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.datasets import cifar10
# from keras.callbacks import TensorBoard
# from keras.metrics import mean_squared_error
# from keras import backend as K
# import numpy as np
# import matplotlib.pyplot as plt

# https://blog.keras.io/building-autoencoders-in-keras.html
# https://elix-tech.github.io/ja/2016/07/17/autoencoder.html

(x_train, _), (x_test, _) = cifar10.load_data()
x_train = x_train.astype('float32') / 255.  # (50000, 32, 32, 3)
x_test = x_test.astype('float32') / 255.

# input_img = Input(shape=(3, 32, 32))  # theano
input_img = Input(shape=(32, 32, 3))  # tensorflow

x = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(input_img)
x = MaxPooling2D((2, 2), border_mode='same')(x)
x = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(x)
encoded = MaxPooling2D((2, 2), border_mode='same')(x)

x = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Convolution2D(3, 3, 3, activation='sigmoid', border_mode='same')(x)

autoencoder = Model(input_img, decoded)

# print(autoencoder.layers)
# print(autoencoder.layers[-1].output_shape)

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

autoencoder.fit(x_train, x_train,
                nb_epoch=50,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))

# If you want to use TensorBoard
# autoencoder.fit(x_train, x_train,
#                 nb_epoch=50,
#                 batch_size=256,
#                 shuffle=True,
#                 validation_data=(x_test, x_test),
#                 callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])

autoencoder.save('autoencoder.cifar10.h5')


### Display decoded images

# decoded_imgs = autoencoder.predict(x_test) # (10000, 32, 32, 3)
#
# n = 10
# plt.figure(figsize=(20, 4))
# for i in range(n):
#     # display original
#     ax = plt.subplot(2, n, i + 1)
#     plt.imshow(x_test[i])
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
#
#     # display reconstruction
#     ax = plt.subplot(2, n, i + 1 + n)
#     plt.imshow(decoded_imgs[i])
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
#
#     loss = mean_squared_error(x_test[i], decoded_imgs[i])
#     print(i, 'loss=', K.eval(loss))
#
# plt.show()
#

### Display weights

# n = 10
# encoder = Model(input_img, encoded)
# encoded_imgs = encoder.predict(x_test[:n])  # (10, 8, 8, 32)
# encoded_imgs = encoded_imgs.reshape(n, 32, 8, 8)  # tensorflow
#
# plt.figure(figsize=(20, 8))
# for i in range(n):
#     for j in range(8):
#         ax = plt.subplot(8, n, j*n + i+1)
#         plt.imshow(encoded_imgs[i][j], interpolation='none')
#         plt.gray()
#         ax.get_xaxis().set_visible(False)
#         ax.get_yaxis().set_visible(False)
# plt.show()
