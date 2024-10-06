import os
import matplotlib.pyplot as plt
import matplotlib.image as mping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mping
from tensorflow import keras
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import keras.layers as layers

plt.figure(figsize=(15,15))
folder ='assets/moto'
imgs = os.listdir(folder)
for i, nameimg in enumerate(imgs[:10]):
  plt.subplot(5,5,i+1)
  img = mping.imread(folder + '/' +nameimg)
  plt.imshow(img)

dataGen= ImageDataGenerator( #instance of the data argumentator, how to randomly change the training data
    rescale =1. /255,
    rotation_range = 30,
    width_shift_range =0.25,
    height_shift_range = 0.25,
    shear_range = 15,
    zoom_range = [0.5,1.5],
    validation_split = 0.2
)

trainingDataSet = dataGen.flow_from_directory('assets',shuffle=True, target_size=(224,224),batch_size=32, subset='training')
#create subsets for training, from directory

evaluateDataSet = dataGen.flow_from_directory('assets',shuffle=True, target_size=(224,224),batch_size=32, subset='validation')






mobileNet = keras.applications.MobileNetV2(include_top=False)


model = keras.Sequential([
    mobileNet,
    layers.Conv2D(filters=1, padding='same',kernel_size=(5),kernel_regularizer=keras.regularizers.L1(0.003)),
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
ep = 50
historial = model.fit(
    trainingDataSet,
    epochs=ep,
    batch_size=32,
    validation_data = evaluateDataSet
)

rangeEpochs = range(ep)

acc = historial.history['accuracy']
val_acc = historial.history['val_accuracy']

loss = historial.history['loss']
val_loss = historial.history['val_loss']

model.save('state.keras')
plt.figure(figsize=(10,10))

plt.subplot(1,2,2)
plt.plot(rangeEpochs, acc, label='precision entrenamiento')
plt.plot(rangeEpochs, val_acc, label='precision validacion')
plt.title('precision')

plt.subplot(1,2,1)
plt.plot(rangeEpochs, loss, label='perdida evaluacion') 
plt.plot(rangeEpochs, val_loss, label='perdida validacion')
plt.title('perdida')

plt.show()
