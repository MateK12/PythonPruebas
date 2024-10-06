import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from plotter import plot_data,plot_scatter
from model import build_model

dataset_path = keras.utils.get_file("auto-mpg.data", "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
dataset_path

column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                'Acceleration', 'Model Year', 'Origin']

raw_dataset = pd.read_csv(dataset_path,names=column_names,
                           comment='\t',na_values='?',
                          sep=' ', skipinitialspace=True)

dataset = raw_dataset.copy()

dataset = dataset.dropna()

origin = dataset.pop('Origin')#transforms the origin column into a one hot vector for each country

dataset['USA'] = (origin==1)*1
dataset['Europe'] = (origin==2)*1
dataset['Japan'] = (origin==3)*1

train_dataset = dataset.sample(frac=0.8,random_state=0)
test_dataset = dataset.drop(train_dataset.index)

# sns.pairplot(train_dataset[['MPG','Cylinders','Displacement','Weight']],diag_kind='kde')
# plt.show()
statistics = train_dataset.describe()
statistics.pop('MPG')
statistics = statistics.transpose()

train_dataset_labels = train_dataset.pop('MPG')
test_dataset_labels = test_dataset.pop('MPG')

def norm(x):
    return (x - statistics['mean']) / statistics['std']

normed_train_DS = norm(train_dataset)
normed_test_DS = norm(test_dataset)



model = build_model(train_dataset)



class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs):
        if epoch % 100 == 0 :print('')
        print('.',end='')

EPOCHS = 1000
earlyStop = keras.callbacks.EarlyStopping(monitor='val_loss',patience=10)

history = model.fit(
    normed_train_DS, train_dataset_labels,
    epochs = EPOCHS, validation_split=0.2,verbose=0,
    callbacks=[PrintDot(),earlyStop]
)

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
print('\n')

plot_data(history)
predictions = model.predict(normed_test_DS).flatten()
plot_scatter(test_dataset_labels,predictions)
