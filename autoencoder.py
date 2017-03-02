from keras.models import load_model
from keras.metrics import mean_squared_error
from keras import backend as K


def autoencoder_model():
    autoencoder = load_model('autoencoder.cifar10.h5')
    return autoencoder


def predict(input_imgs):  # (32, 32, 3)
    autoencoder = autoencoder_model()
    decoded_imgs = autoencoder.predict(input_imgs)
    losses = [K.eval(mean_squared_error(i_img, o_img)) for (i_img, o_img) in zip(input_imgs, decoded_imgs)]
    return decoded_imgs, losses
