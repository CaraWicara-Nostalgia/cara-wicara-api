import os
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import TFSMLayer
from tensorflow.keras.applications.efficientnet import preprocess_input
import numpy as np

def display_results(img_path, predictions, labels):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.imshow(img)
    plt.axis('off')

    predicted_label = labels[np.argmax(predictions)]
    confidence = predictions[0][np.argmax(predictions)] * 100

    plt.title(f'Prediction: {predicted_label}\nConfidence: {confidence:.2f}%')
    plt.show()
    
    return predicted_label, confidence

def preprocess_input_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))  # Resize image to the input size expected by the model
    img = preprocess_input(img)  # Preprocess the image as per the model's requirements
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

def predict_image(image_path):
    # List of class labels
    class_labels = ['Heran', 'Sedih', 'Senang', 'Tenang', 'Tertawa']
    
    model_path = "model_test.h5"  
    
    # # Load the model from the .h5 file
    # model_path = "..---------------------------------------------------------------------------------==
    # # Verify if the model file exists
    # if not os.path.exists(model_path):
        # raise FileNotFoundError(f"Model file not found at {model_path}. Please check the path.")
    
    model = load_model(model_path)
    prediction = model.predict(preprocess_input_image(image_path))

    # Display the results
    predicted_label, confidence_value = display_results(image_path, prediction, class_labels)
    
    return predicted_label, confidence_value
