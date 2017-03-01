import tensorflow as tf
import keras
from flask import Flask

app = Flask(__name__)


@app.route('/')
def index():
    # return tf.__version__
    return keras.__version__


if __name__ == '__main__':
    app.debug = True
    app.run()
