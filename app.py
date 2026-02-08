import os
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Load trained model
model = load_model("model/pulmo_model.h5")

# Class order MUST match training
class_names = ['COVID', 'Normal', 'Pneumonia', 'Tuberculosis']

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None

    if request.method == 'POST':
        file = request.files['file']
        filepath = os.path.join("static", file.filename)
        file.save(filepath)

        img = image.load_img(filepath, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        preds = model.predict(img_array)
        prediction = class_names[np.argmax(preds)]

    return render_template(
        'index.html',
        prediction=prediction
    )

if __name__ == '__main__':
    app.run(debug=True)
