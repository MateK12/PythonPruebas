import torch
import torch.nn as nn
import torchvision
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

batch_size = 64
learning_rate = 0.0001
num_epochs = 20
WD = 0.00002

allTransforms = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)) 
])

train_Dataset = torchvision.datasets.CIFAR10(
    root='./data',download=False,transform=allTransforms,train=True
)
test_dataset = torchvision.datasets.CIFAR10(root='./data',download=True,transform=allTransforms,train=True)

train_dataLoader = DataLoader(train_Dataset,batch_size=batch_size,shuffle=True)
test_datasetLoader = DataLoader(test_dataset,batch_size=batch_size,shuffle=True)

class Identity(nn.Module): # a layer that will be used to replace the useless layers in our tuned model
    def __init__(self):
        super(Identity,self).__init__()
    def forward(self,x): # returns the input untouched
        return x #f(x)=x
    
model = torchvision.models.vgg16(pretrained=True)

lossfn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model.parameters(),lr=learning_rate, weight_decay=WD)

model.avgpool = Identity()
model.classifier = nn.Linear(in_features=512,out_features=10)

print(model)

for epoch in range(num_epochs):
    for i, (images,labels) in enumerate(train_dataLoader):
        prediction = model(images)
        loss = lossfn(prediction, labels)


        optimizer.zero_grad()
        optimizer.step()
        loss.backward()
    print("The loss in the epoch {} is {:.4f}".format(epoch+1,num_epochs,loss.item()))

with torch.no_grad():
    correct = 0 
    total = 0 
    for images, labels in train_dataLoader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    print('Accuracy of the network on the {} train images: {} %'.format(50000, 100 * correct / total))