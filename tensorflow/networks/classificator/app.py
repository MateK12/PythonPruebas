from flask import Flask, render_template,request, redirect
from tensorflow import keras
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import keras.layers as layers
from PIL import Image
from io import BytesIO
import cv2

app = Flask(__name__)
mobileNet = keras.applications.MobileNetV2(include_top=False)
model = keras.Sequential([
        mobileNet,
        layers.Conv2D(filters=1,name='primer_conve', padding='same',kernel_size=(5),kernel_regularizer=keras.regularizers.L1(0.003)),
        layers.MaxPooling2D(2,2),
        layers.Dropout(0.5),
        layers.Flatten(),
        layers.Dense(units=3, activation='softmax')
    ])

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
 
    file =request.files['file']
    model.load_weights('/home/mateo/Escritorio/fintech/dataManager/networks/classificator/assets/new_weights.weights.h5')
    print('weights loaded')


    pic = Image.open(file)
    pic = np.array(pic).astype(float) /255
    pic = cv2.resize(pic,(224,224))
    pic = pic.reshape(-1, 224, 224, 3)

    print('fdsfs')
    prediction = model.predict(pic)
    print(prediction[0])
    if np.argmax(prediction[0]) == 0:
        print ("moto")  
    elif np.argmax(prediction[0]) == 1:
        print ("auto")
    elif np.argmax(prediction[0]) == 2:
        print ("avion")

    # file.save(f'{file.filename}')
    return redirect('/')





if __name__ == '__main__':
    app.run(debug=True, port=5000)