import tensorflow as tf 
from tensorflow.keras.models import load_model

model = load_model('./model/model_test.h5')

print(model)