import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from matplotlib import pyplot

batch_size = 64
num_classes = 10
learning_rate = 0.001
num_epochs = 20
loaded = True

lossValues=[]
accValues=[]



allTransforms = transforms.Compose([#method to encapsulate data augmentation logic in one single variable
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)) #mean and standard deviation for RGB channels
])

TrainDataset = torchvision.datasets.CIFAR10(
    root='./data',
    train=True,
    transform=allTransforms,
    download=True,
) 

TestDataset = torchvision.datasets.CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform=allTransforms,)

trainDataLoader= DataLoader(dataset=TrainDataset,batch_size=batch_size, shuffle=True) #instantiate DataLoaders
testDataLoader = DataLoader(dataset=TestDataset,batch_size=batch_size, shuffle=True)

class CNN(nn.Module): #neural Network extends nn.module
    def __init__(self,classes) : #initialize nn.Module()
        super(CNN,self).__init__()
        self.conv_layer1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3,padding="same")
        self.conv_layer2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3,padding="same") 
        self.maxPool_layer1 = nn.MaxPool2d(kernel_size=2,stride=2)  

        self.conv_layer3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3,padding="same")
        self.conv_layer4 = nn.Conv2d(in_channels=64,out_channels=64, kernel_size=3,padding="same")
        self.maxPool_layer2 = nn.MaxPool2d( kernel_size=2, stride=2)
        
        self.conv_layer4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3,padding="same")
        self.conv_layer5 = nn.Conv2d(in_channels=64,out_channels=128, kernel_size=3,padding="same")
        self.maxPool_layer3 = nn.MaxPool2d(kernel_size=2,stride=2)

        self.fc1 = nn.Linear(in_features=2048,out_features=128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=128,out_features=classes)

    def forward(self, input): #input flow through the network
        out = self.conv_layer1(input)
        out = self.conv_layer2(out)
        out = self.maxPool_layer1(out)

        out = self.conv_layer3(out)
        out = self.conv_layer4(out)
        out = self.maxPool_layer2(out)
        
        out = self.conv_layer4(out)
        out = self.conv_layer5(out)
        out = self.maxPool_layer3(out)

        out = out.reshape(out.size(0), -1)

        out = self.fc1(out)
        out = self.relu1(out)
        prediction = self.fc2(out)
        return prediction

model = CNN(num_classes)

lossFn = nn.CrossEntropyLoss()

optim = torch.optim.SGD(model.parameters(),lr=learning_rate,momentum=0.9,weight_decay=0.005)

def save_checkpoint(state,fileName):
    print('saving model')
    torch.save(state, fileName)
def load_checkpoint(state):
    print('loading model')
    model.load_state_dict(state['state_dict']) # load the parameters in our model 
    optim.load_state_dict(state['optimizer_state_dict'])

if loaded:
    load_checkpoint(torch.load('my_checkpoint_at_19thEpoch.pth.tar')) #with torch.load, we load the state


for epoch in range(num_epochs):
    # if epoch ==19:
    #     checkpoint = {"sate_dict":model.state_dict(),"optimizer":optim.state_dict()}
    #     save_checkpoint({ # save checkpoint dictionary
    #         'epoch':epoch +1,
    #        'state_dict': model.state_dict(), #current values of weights and biases
    #         'optimizer_state_dict': optim.state_dict(), #current state of the optimizer
    #     },'my_checkpoint_at_19thEpoch.pth.tar')

    for i ,(image, label) in enumerate(trainDataLoader):
        
        predictions = model(image)
        loss = lossFn(predictions,label)

        # Backward and optimize
        optim.zero_grad()
        loss.backward()
        optim.step()
    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
    lossValues.append(loss.item())
    print(lossValues)
    print(type(lossValues))
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in trainDataLoader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    print('Accuracy of the network on the {} train images: {} %'.format(50000, 100 * correct / total))

fig, plotter = pyplot.subplots()

plotter.set_ylabel = 'loss on training'
plotter.set_xlabel = 'Epoch'
plotter.set_title  = 'Accuracy'
plotter.plot(range(num_epochs),lossValues)
plotter.legend()
pyplot.show()