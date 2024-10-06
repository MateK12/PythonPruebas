from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout, BatchNormalization, Input
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint
import os
import numpy as np
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.applications.vgg16 import VGG16
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import keras.layers as layer
import keras.models as model
import keras
import tensorflowjs as tfjs


epochs = 5
width_shape = 224
heigth_shape = 224
num_classes = 3
batchSize = 32

train_data_path = 'assets'

dataGen= ImageDataGenerator(
    rescale =1. /255,
    rotation_range = 30,
    width_shift_range =0.25,
    height_shift_range = 0.25,
    shear_range = 15,
    zoom_range = [0.5,1.5],
    validation_split = 0.2
)

trainingDataSet = dataGen.flow_from_directory('assets',shuffle=True, target_size=(224,224),batch_size=32, subset='training')


evaluateDataSet = dataGen.flow_from_directory('assets',shuffle=True, target_size=(224,224),batch_size=32, subset='validation')


@keras.saving.register_keras_serializable(package="my_custom_package")
class VGG16_Layer(layer.Layer):
    def __init__(self):
        super(VGG16_Layer, self).__init__()
        self.vgg16 = VGG16(include_top=False, weights='imagenet', input_shape=(224,224,3))
    def call(self,tensor):
        x= self.vgg16(tensor)
        return x 
    
@keras.saving.register_keras_serializable(package="my_custom_package")
class Clasification(layer.Layer):
    def __init__(self):
        super(Clasification, self).__init__()
        self.flatten = Flatten()
        self.dense = Dense(units=3, activation='softmax')
    def call(self,tensor):
        x = self.flatten(tensor)
        x = self.dense(x)
        return x
    
@keras.saving.register_keras_serializable(package="my_custom_package")
class Conv_Model(Model):
    def __init__(self):
        super(Conv_Model,self).__init__()
        self.transfer = VGG16_Layer()
        self.classification = Clasification()
    def call(self,tensor):
        x = self.transfer(tensor)
        x = self.classification(x)
        return x 
model = Conv_Model()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(trainingDataSet,epochs=epochs,validation_data=evaluateDataSet)


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

model.save_weights('assets/MLOPS_weights')
loaded_model = keras.models.load_model('assets/MLOPS.keras')
# tfjs.converters.save_keras_model(loaded_model, 'jsModel')


fig, ax = plt.subplots() 
rangeEpochs =range(epochs)
ax.plot(rangeEpochs, acc, label='acierto en entrenamiento', color='red')
ax.plot(rangeEpochs, val_acc, label='acierto en validacion', color='blue')
ax.set_title('Aciertos')
ax.set_xlabel('Epocas',loc='right')
ax.set_ylabel('valores')
ax.legend( fontsize='large')


fig, lss = plt.subplots() 


lss.plot(rangeEpochs, loss, label='perdida en entrenamiento', color='red')
lss.plot(rangeEpochs, val_loss, label='perdida en validacion', color='blue')
lss.set_title('Perdida')
lss.set_xlabel('Epocas')
lss.set_ylabel('valores')


plt.show()