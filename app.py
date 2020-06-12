
def analyze():
    from tensorflow.keras.preprocessing.image import load_img
    from tensorflow.keras.preprocessing.image import img_to_array
    import time
    from tensorflow.keras.applications.vgg16 import VGG16
    from tensorflow.keras.applications.vgg16 import preprocess_input
    import numpy as np
    from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions

    model = VGG16(
        weights=None,
        include_top=True,
        classes=10,
        input_shape=(32, 32, 3)
    )

    model.compile(
        loss='categorical_crossentropy',
        optimizer='sgd',
        metrics=['accuracy']
    )

    # load the model
    model.load_weights('model.h5')

    from tensorflow.keras.preprocessing.image import load_img
    from tensorflow.keras.preprocessing.image import img_to_array
    from tensorflow.keras.applications.vgg16 import preprocess_input

    # load an image from file
    image = load_img('static/img.png', target_size=(32, 32))
    # convert the image pixels to a numpy array
    image = img_to_array(image)
    # reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # prepare the image for the VGG model
    image = preprocess_input(image)
    # predict the probability across all output classes
    yhat = model.predict(image)

    data = np.array2string(yhat)

    datareplace = str(data).replace('[', '')
    datareplace2 = str(datareplace).replace(']', '')

    values = datareplace2.split()

    spot = values.index(max(values))

    if spot == 0:
        classification = 'airplane'
    elif spot == 1:
        classification = 'automobile'
    elif spot == 2:
        classification = 'bird'
    elif spot == 3:
        classification = 'cat'
    elif spot == 4:
        classification = 'deer'
    elif spot == 5:
        classification = 'dog'
    elif spot == 6:
        classification = 'frog'
    elif spot == 7:
        classification = 'horse'
    elif spot == 8:
        classification = 'ship'
    elif spot == 9:
        classification = 'truck'
    else:
        classification = 'error'

    return classification



from flask import Flask, render_template, redirect, url_for, request

# Route for handling the login page logic
app = Flask(__name__)

import os
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'static/'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])

def upload_file():
    classification = analyze()
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = "img.png"
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('upload_file',
                                    filename=filename))
    return render_template('index.html', classification=classification)