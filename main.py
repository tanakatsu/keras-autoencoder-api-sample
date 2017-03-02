from flask import Flask, request, render_template
import json
import imageutil
import autoencoder
import tensorflow as tf
import keras


app = Flask(__name__)

IMAGE_WIDTH = 32
IMAGE_HEIGHT = 32


def predict_image(img):
    resized = img.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
    input_img = imageutil.to_np_data_array(resized)
    input_img = input_img.reshape(-1, IMAGE_HEIGHT, IMAGE_WIDTH, 3)  # (32, 32, 3) -> (1, 32, 32, 3)
    images, losses = autoencoder.predict(input_img)
    return imageutil.to_pil_image(images[0]), float(losses[0])


@app.route('/')
def index():
    print(tf.__version__)
    print(keras.__version__)
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        url = request.args.get('url')
        data = request.args.get('data')
        if data:
            img = imageutil.read_image_from_base64(data)
        elif url:
            img = imageutil.read_image_from_url(url)
        else:
            return '''
            <h3>Error: Invalid parameter</h3>
            '''
        outimg, loss = predict_image(img)
        return render_template('result.html', loss=loss, imgin=imageutil.encode_base64(img), imgout=imageutil.encode_base64(outimg))
    else:
        f = request.files['file']
        img = imageutil.read_image_from_file(f)
        outimg, loss = predict_image(img)
        return render_template('result.html', loss=loss, imgin=imageutil.encode_base64(img), imgout=imageutil.encode_base64(outimg))


@app.route('/predict.json', methods=['GET', 'POST'])
def predict_json():
    if request.method == 'GET':
        url = request.args.get('url')
        data = request.args.get('data')
        if data:
            img = imageutil.read_image_from_base64(data)
        elif url:
            img = imageutil.read_image_from_url(url)
        else:
            return json.dumps({'error': 'Invalid parameter'})

        outimg, loss = predict_image(img)
        return json.dumps({'loss': loss})
    else:
        f = request.files['file']

        img = imageutil.read_image_from_file(f)
        outimg, loss = predict_image(img)
        return json.dumps({'loss': loss})


@app.route('/upload')
def upload():
    return render_template('upload.html')


if __name__ == '__main__':
    app.debug = True
    app.run()
