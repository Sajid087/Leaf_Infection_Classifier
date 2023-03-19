from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from typing import OrderedDict
from io import BytesIO
from PIL import Image
import numpy as np


app = Flask(__name__)

# Load the saved model
model = tf.keras.models.load_model('E:\\coding\\pdr project\\plantdiseaserecognition\\best_accmodel.h5')

# Define a home page route
@app.route('/')
def home():
    return render_template('home.html')

def read_file_as_image(image) -> np.ndarray:
    image =np.array(Image.open(BytesIO(image)))
    return image

# Define a prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    # Get the request data
    image = request.files['image'].read()
    
    # Preprocess the data
    # ...

    # Convert the image bytes to a PIL image object
    image =np.array(Image.open(BytesIO(image)))

    # Conver the PIL image object to a numpy array
    image = np.array(image)
    
    # Make a prediction
    inputs = tf.constant(OrderedDict({'image': [image]}))
    outputs = model(inputs)
    prediction = tf.argmax(outputs['output'], axis=1).numpy()[0]
    
    # Postprocess the prediction
    # ...
    
    # Return the prediction
    return jsonify({'prediction': prediction})

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
