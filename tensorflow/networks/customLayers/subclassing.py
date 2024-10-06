import tensorflow as tf
from tensorflow import keras
import numpy as np
import keras.layers as layers

from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(-1, 28,28,1).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28,28,1).astype('float32') / 255.0


results = []

class CNN_block(layers.Layer): #the class implements the interface layer
    def __init__(self, filters, kernel_size=3): 
        super(CNN_block, self).__init__() #runs the constructor of the superclass (keras.layers)
        self.conv = layers.Conv2D(filters, kernel_size,padding='same') #3 layers convolution, batchNormalization, and maxpooling
        self.batchnorm = layers.BatchNormalization()
        self.maxpool = layers.MaxPool2D()
    def call(self, tensor, training=False): # method for executing the block 
        x = self.conv(tensor, training=training)
        # print('debug whatewer you want')
        x = self.batchnorm(tensor, training=training)
        x = tf.nn.relu(x)
        return x        
    
class Clasification(layers.Layer):
    def __init__(self,units=10):
        super(Clasification, self).__init__()
        self.flatten = layers.Flatten()
        self.dense = layers.Dense(units=units, activation='softmax')
    def call(self,tensor):
        x = self.flatten(tensor)
        x = self.dense(x)
        return x


class Model_structure(keras.Model): # implements the model interface
    def __init__(self):#define the structure of the model, like model.Sequential([Model_structure])
        super(Model_structure, self).__init__() #define the layers that will be used
        self.block1 = CNN_block(32)
        self.block2 = CNN_block(64)
        self.block3 = CNN_block(128)
        self.clasification = Clasification()
    def call(self, tensor1): #define how will be executed
        x = self.block1(tensor1) 
        x = self.block2(x)
        x = self.block3(x)
        x = self.clasification(x)
        return x


model = Model_structure()



model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

train_data = model.fit(x_train, y_train,batch_size=64, epochs=3, verbose=2)
model.evaluate(x_test, y_test, batch_size=64)

for i in range(0,3):
    inputTensor = np.random.randn(1,28,28,1)
    inputTensor = inputTensor.reshape(-1,28,28,1)
    output = model(inputTensor)
    print(output.numpy()[0][4])
    param =  list(output.numpy()[0])
    index = np.argmax(param)
    print(type(index))
    results.append(int(index))


print(results)