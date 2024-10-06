from flask import Flask, render_template,request, redirect
import keras
from keras.applications.imagenet_utils import preprocess_input
from PIL import Image
from io import BytesIO  
import cv2
import numpy as np
from keras.models import load_model
import keras.layers as layer
from keras.layers import  Dense, Flatten
from keras.applications.vgg16 import VGG16
from keras.models import Sequential, Model


    
print('loading model')
model =keras.models.load_model('assets/MLOPS.keras')
print('model loaded')

app = Flask(__name__)

@app.route('/',methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file =request.files['file']
    print(file.filename)

    pic = Image.open(file)
    pic = np.array(pic).astype(float) /255
    pic = cv2.resize(pic,(224,224))
    pic = pic.reshape(-1, 224, 224, 3)
    print(model.predict(pic))
    return redirect('/')


if __name__ == '__main__':
    app.run(debug=True, port=5000)