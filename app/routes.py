from app import app
from flask import jsonify

@app.route('/')
def root():
    return 

@app.route('/index', methods=['get'])
def index():
    data = {
        "message":"success",
        "status":200,
        "data": {}
    }
    return jsonify(data), 400