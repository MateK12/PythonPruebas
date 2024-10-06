import tensorflow as tf
import keras
import keras.layers as layers
import numpy as np
from ucimlrepo import fetch_ucirepo #library from the UCI (university of California Irvine)
import os
from matplotlib import pyplot as plt
import pandas as pd
from keras import regularizers



os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# fetch dataset 
student_performance = fetch_ucirepo(id=320) 
features = student_performance.data.features 

guardian=features.pop('guardian')

features['sex'] = (features['sex']=='F')*1
features['romantic'] = (features['romantic']=='yes')*1
features['address'] = (features['address']=='U')*1
features['famsize']=(features['famsize']=='GT3')*1
features['Pstatus']=(features['Pstatus']=='T')*1
features['guardianF'] = (guardian=='father')*1
features['guardianM'] = (guardian=='mother')*1
features['guardianOther'] = (guardian=='other')*1
ndArrayFeatures = features.to_numpy()
for f in range(31):
    if ndArrayFeatures[:,f][0] == 'yes' or ndArrayFeatures[:,f][0] == 'no':
        ndArrayFeatures[:,f] = (ndArrayFeatures[:,f]=='yes')*1
    if isinstance(ndArrayFeatures[:,f][0],str):
        ndArrayFeatures[:,f] = pd.Categorical(ndArrayFeatures[:,f]).codes #assigns a nu

depuredFeatures = pd.DataFrame(ndArrayFeatures).astype(float)

# features = depuredFeatures.select_dtypes(exclude=['object']) #excludes string columns
features = depuredFeatures
features = np.asarray(features).astype('float32')
targets= student_performance.data.targets 
EPOCHS = 700

train_DS = features.copy()

#print(student_performance.dtypes) #shows the type of data of each column (only for pd's dataframes)


model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=features[0].shape),
    layers.Dense(64, activation='relu',kernel_regularizer=regularizers.L1(1e-4)),
    layers.Dense(64, activation='relu',kernel_regularizer=regularizers.L1(1e-4)),
    layers.Dense(32, activation='relu'),
    layers.Dense(12, activation='relu'),
    layers.Dense(3) 
])

model.compile(loss='mae',optimizer='adam',metrics=['accuracy'])

class dotCallback(keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs):
        if epoch % 100 == 0 :print('')
        print('.',end='')

history = model.fit(
    x=train_DS,
    y=targets,
    epochs = EPOCHS,
    validation_split=0.2,
    verbose=0,
    callbacks=[dotCallback()]
)

fig, axis = plt.subplots()
rangeEpochs= range(EPOCHS)
fig.suptitle('PERDIDA')
axis.set_xlabel('epocas')
axis.set_ylabel('pérdida')
axis.plot(rangeEpochs, history.history['loss'], label='Pérdida en entrenamiento')
axis.plot(rangeEpochs, history.history['val_loss'], label='Pérdida en validación')
axis.legend()

figAcc, axisAcc = plt.subplots()
figAcc.suptitle('EFECTIVIDAD')

axisAcc.set_xlabel('epocas')
axisAcc.set_ylabel('eficacia')
axisAcc.plot(rangeEpochs,history.history['accuracy'], label='efectividad en entrenamiento')
axisAcc.plot(rangeEpochs,history.history['val_accuracy'], label='efectividad en validacion')
axisAcc.legend()

plt.show()