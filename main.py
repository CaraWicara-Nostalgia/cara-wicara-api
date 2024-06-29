import os
import tensorflow as tf
import numpy as np
import cv2
import tensorflow as tf
from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input

app = Flask(__name__)


# Define constants for valid extensions and upload folder
VALID_EXTENSIONS = {'png', 'jpg', 'jpeg'}
FOLDER_UPLOAD = 'uploads/'

model_path = 'model/model_test.h5'
model = load_model(model_path, compile=False)

# List of class labels
class_labels = ['Heran', 'Sedih', 'Senang', 'Tenang', 'Tertawa']

def validExtension(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in VALID_EXTENSIONS 
    

def preprocess_input_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

@app.route('/')
def root():
    try:
        data = {
            "message":"Service API run properly",
            "status":200,
        }
        return jsonify(data), 200
    except Exception as e: 
        print('error: ', e)
        return jsonify({"message": "Internal Server Error", "status": 500,}), 500

@app.route('/index', methods=['GET'])
def index():
    try:
        data = {
            "message":"success",
            "status":200,
            "data": {}
        }
        return jsonify(data), 200
    except Exception as e: 
        print('error: ', e)
        return jsonify({"message": "Internal Server Error", "status": 500,}), 500

@app.route('/predict', methods=['POST'])
def upload():
    try: 
        # Check if the request has an image file
        if 'image' not in request.files:
            return jsonify({ 
                "data":null, 
                "message":"image required!", 
                "status":400 
            }), 400

        
        image_file = request.files['image']
        if image_file.filename == "" or not validExtension(image_file.filename):
            return jsonify({
                "data":null, 
                "message":"extension image is not valid!",
                "status":400
            }), 400
        
        filename = secure_filename(image_file.filename)
        image_path = os.path.join(FOLDER_UPLOAD, filename)
        image_file.save(image_path)
        
        # Preprocess the image
        preprocessed_image = preprocess_input_image(image_path)

        # Make a prediction
        predictions = model.predict(preprocessed_image)
        
        # Determine the predicted label and confidence
        predicted_label = class_labels[np.argmax(predictions)]
        confidence = predictions[0][np.argmax(predictions)] * 100
        
        # Return the result as JSON
        return jsonify({
            "data":{
                'prediction': predicted_label, 
                'confidence': confidence
            }, 
            "message":"prediction is sucessfully",
            "status":200
        }), 200
    
    except Exception as e:
        print('error: ', e)
        print(e)
        return jsonify({"message": 'internal server error', "status": 500,}), 500

if __name__ == '__main__':
    app.run(debug=True)