from flask import Flask, jsonify

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'