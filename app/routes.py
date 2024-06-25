from app import app
from flask import jsonify

@app.route('/')
def root():
    return 

@app.route('/index', methods=['get'])
def index():
    try:
        data = {
            "message":"success",
            "status":200,
            "data": {}
        }
        asd
        return jsonify(data), 400
    except Exception as e: 
        print('error: ', e)
        return jsonify({"message": "Internal Server Error", "status": 500,}), 500