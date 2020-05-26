#Usage: python app.py
import os
 
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential
import numpy as np
import argparse
import imutils
import cv2
import time
import uuid
import base64
import tensorflow as tf 
from keras import backend as K
import keras
import sys
import matplotlib.pyplot as plt
from PIL import Image

img_width, img_height = 28, 28
model_path = './model/model.hdf5'
model =  tf.keras.models.load_model(model_path)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpeg', 'jpg', 'pdf'])

def get_as_base64(url):
    return base64.b64encode(request.get(url).content)

def predict(file):
    x = load_img(file, target_size=(img_width,img_height))
    x = img_to_array(x).astype('float16')/255
    x = np.expand_dims(x, axis=0)
    array = model.predict(x)
    result = array[0]
    answer = np.argmax(result)
    if answer == 0:
        print("Label: Bacterial Pneumonia")
    elif answer == 1:
	    print("Label: Covid-19")
    elif answer == 2:
	    print("Label: Normal")
    elif answer == 3:
	    print("Label: Viral Pneumonia")
    session = keras.backend.get_session()
    init = tf.compat.v1.global_variables_initializer()
    session.run(init)

    class_output = model.output[:, answer]
    last_conv_layer = model.get_layer('conv2d_15')
    class_output = model.output[:, answer]
    grads = K.gradients(class_output, last_conv_layer.output)[0]
    pooled_grads = K.mean(grads, axis=(0, 1, 2))
    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
    pooled_grads_value, conv_layer_output_value = iterate([x])
    for i in range(128):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
    heatmap = np.mean(conv_layer_output_value, axis=-1)
    for x in range(heatmap.shape[0]):
        for y in range(heatmap.shape[1]):
            heatmap[x,y] = np.max(heatmap[x,y],0)

    heatmap = np.maximum(heatmap,0)
    heatmap /= np.max(heatmap)
    plt.imshow(np.real(heatmap))

    x = load_img(file, target_size=(28,28))
    import skimage
    upsample = skimage.transform.resize(heatmap, (28,28),preserve_range=True)
    plt.imshow(x)
    plt.imshow(upsample,alpha=0.5)
    plt.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off')
    plt.savefig('./uploads/foo.png', bbox_inches='tight', pad_inches = 0, transparent = True)
    
    image = Image.open('./uploads/foo.png')
    image.thumbnail((1024, 1024))
    image.save('./uploads/foo.png')
    return answer

def my_random_string(string_length=10):
    """Returns a random string of length string_length."""
    random = str(uuid.uuid4()) # Convert UUID format to a Python string.
    random = random.upper() # Make all characters uppercase.
    random = random.replace("-","") # Remove the UUID '-'.
    return random[0:string_length] # Return the random string.

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/")
def template_test():
    return render_template('template.html', label='', imagesource='../uploads/person1_bacteria_1.jpeg')

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        import time
        start_time = time.time()
        file = request.files['file']

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)

            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            result = predict(file_path)
            if result == 0:
                label = "Bacterial Pneumonia"
            elif result == 1:
                label = "Covid-19"
            elif result == 2:
                label = "Normal"
            elif result == 3:
                label = "Viral Pneumonia"
            print(result)
            print(file_path)
            filename = my_random_string(6) + filename

            os.rename(file_path, os.path.join(app.config['UPLOAD_FOLDER'], filename))
            print("--- %s seconds ---" % str (time.time() - start_time))

            return render_template('template.html', label=label, imagesource='./uploads/foo.png')#'../uploads/' + filename)

from flask import send_from_directory

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

from werkzeug.middleware.shared_data import SharedDataMiddleware
app.add_url_rule('/uploads/<filename>', 'uploaded_file',
                 build_only=True)
app.wsgi_app = SharedDataMiddleware(app.wsgi_app, {
    '/uploads':  app.config['UPLOAD_FOLDER']
})

if __name__ == "__main__":
    app.debug=False
    app.run(host='0.0.0.0', port=3000)
