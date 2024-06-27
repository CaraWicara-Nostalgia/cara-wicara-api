from flask import request, jsonify
from werkzeug.utils import secure_filename
from handlers.prediction_model import predict_image
from app import app
import os

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_files(filename):
    return '.' in filename and \
    filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    

def upload_handler():
    try: 
        if 'image' not in request.files:
            return jsonify({"message":"image not found", "status":400}), 400
        
        image = request.files['image']
    
        if image.filename == '':
            return jsonify({"message": "No selected file", "status": 400}), 400


        if image and allowed_files(image.filename):
            filename = secure_filename(image.filename)    
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image.save(filepath)
            print('file uploaded successfully')
            print('file name: ', filename)
            print('file path: ', filepath)
            
        predicted_label, confidence_value = predict_image(f'../uploads/{filename}')
        
        return jsonify({"message": "File uploaded successfully", "status": 200, "data": { "file":image.filename }, "path":filepath}), 200
        
    except Exception as e: 
        print('error: ', e)
        return jsonify({"message": e, "status":500}), 500