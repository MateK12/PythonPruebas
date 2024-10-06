import torch
import torch.nn as nn # the class that contains everything for creating a neural network layer
import torch.optim as optim #the classs with all the optimizers
import torch.nn.functional as F # activation functions
from  torch.utils.data import DataLoader #Dtaloader
import torchvision.datasets as dataset
import torchvision.transforms as transforms

class NN(nn.Module):
    def __init__(self,input_size,num_classes): #defines every layer, and initializes the parent aswell
        super(NN,self).__init__()
        self.layer1 = nn.Linear(input_size, 50) #Linear(input_size, output_size), must match with the next anf previous layer
        self.layer2 = nn.Linear(50,num_classes)
    def forward(self,x):#Define the computation performed at every call.
        x = F.relu(self.layer1(x)) #(x is our features passing trhough every layer)
        x = self.layer2(x)
        return x

#build each part of the network at a time

input_size = 784 #Number of pixels of each number (28x28)
num_classes = 10
batch_size = 64
learning_rate = 0.0001
num_epochs = 1


model = NN(784,10)
x = torch.rand(64,784)
print(model(x).shape)

train_dataset = dataset.MNIST(root='Datasets/',train=True,transform=transforms.ToTensor(),download=True) #first inport the dataset
train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True) #then load it (while transforming and shuffling ti)

test_dataset = dataset.MNIST(root='Datasets/',train=True,transform=transforms.ToTensor(),download=True) #the same for validation
test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=True)


loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate) #model.parameters() (returns an iterated object with the indexed network's biases and weights to) 

for epoch in range(num_epochs):
    for batch_i, (data,labels) in enumerate(train_loader):

        data = data.reshape(data.shape[0],-1)
        
        scores = model(data)
        loss = loss_function(scores,labels) #returns the entropy between the predicted values and the teargets 

        optimizer.zero_grad() #clears the gradients from the last step
        loss.backward() #computes the gradients of the loss  the model parameters (backpropagation)

        optimizer.step() #updates the model parameters using the gradients and the optimizer

def check_values(loader, model):
    num_correct = 0
    num_samples = 0 
    model.eval() #model starts to evaluate

    with torch.no_grad(): # avoids the program to run more gradiants
        for x, y in loader: #x = predicted y=targets

            x = x.reshape(x.shape[0], -1)  #Flatten() 

            scores = model(x) #
            _, predictions = scores.max(1) #get the max output (prediction of the model)

            num_correct += (predictions == y).sum() #add all the correct predictions up
            num_samples += predictions.size(0)

        print(f'got {num_correct}/{num_samples}, accuracy: {(float(num_correct)/float(num_samples))*100:.2f}')
        model.train() 


check_values(train_loader, model)
check_values(test_loader, model)