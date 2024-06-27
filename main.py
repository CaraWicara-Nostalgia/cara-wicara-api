from flask import Flask, jsonify, request
# from handlers.upload_handler import upload_handler
# from tensorflow.keras.models import load_model
from tensorflow import keras
from tensorflow.keras.applications.efficientnet import preprocess_input
import numpy as np
import cv2

app = Flask(__name__)

model_path = 'model/model_test.h5'
model = keras.models.load_model(model_path)

# List of class labels
class_labels = ['Heran', 'Sedih', 'Senang', 'Tenang', 'Tertawa']

def preprocess_input_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))  # Resize image to the input size expected by the model
    img = preprocess_input(img)  # Preprocess the image as per the model's requirements
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

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

@app.route('/index', methods=['get'])
def index():
    try:
        app.logger.info('Hello, logging!')
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
        if 'image' not in request.files:
           return jsonify({'error': 'No image file provided'}), 400
    
        image_file = request.files['image']
        image_path = f"/tmp/{image_file.filename}"
        image_file.save(image_path)
        
        # Preprocess the image
        preprocessed_image = preprocess_input_image(image_path)
        
        # Make a prediction
        predictions = model.predict(preprocessed_image)
        
        # Determine the predicted label and confidence
        predicted_label = class_labels[np.argmax(predictions)]
        confidence = predictions[0][np.argmax(predictions)] * 100
        
        # Return the result as JSON
        return jsonify({'prediction': predicted_label, 'confidence': confidence}), 200
    
    except Exception as e:
        print('error: ', e)
        return jsonify({"message": 'internal server error', "status": 500,}), 500


    
