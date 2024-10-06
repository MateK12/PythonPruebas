import torch.nn as nn
import torch
from torchvision import transforms
import torchvision
from torch.utils.data import DataLoader, Dataset, random_split
from utils.custom_dataset import planes_cars_motos_Dataset
from utils.plotter import Plot_data

import numpy as np

batch_size = 1
learning_rate = 0.0007
epochs = 15    


alltransforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224,224)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = planes_cars_motos_Dataset('/home/mateo/Escritorio/Python/torch/nets/Planes_cars_motos/assets/planes_cars_motos.csv','/home/mateo/Escritorio/Python/torch/nets/Planes_cars_motos/assets/images_augmented',transform=alltransforms)

print('images amount:{}'.format(len(dataset)))

train_dataset, test_dataset = random_split(dataset,[5886,654])

train_dataLoader = DataLoader(train_dataset, batch_size=batch_size,shuffle=True)
test_dataLoader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


print('dataloaders and datasets loaded successfully')

class Identity(nn.Module):
    def __init__(self):
        super(Identity,self).__init__()
    def forward(self, x):
        return x
class classifier_layer(nn.Module):
    def __init__(self):
        super(classifier_layer,self).__init__()
        self.flattener_1 = nn.Flatten()
        self.relu_1 = nn.ReLU()
        self.fc_1 = nn.Linear(25088, 500)
        self.dropOut_1 = nn.Dropout(p=0.3)
        self.fc_2 = nn.Linear(500,3)

    def forward(self, x):
        x = self.flattener_1(x)
        x = self.relu_1(x)
        x = self.fc_1(x)
        x = self.dropOut_1(x)
        x = self.fc_2(x)
        # x,i = torch.max(x,dim=1,)
        return x


model = torchvision.models.vgg16(pretrained=True)
for param in model.parameters():
    param.requires_grad = False


model.avgpool = Identity()
model.classifier = classifier_layer()
lossfn = nn.CrossEntropyLoss()
optim = torch.optim.SGD(model.parameters(), lr=learning_rate,momentum=0.8, weight_decay=0.005)

acc = []

model.train()
for epoch in range(epochs):
    if epoch ==14:
        modeldict = {
        'state_dict': model.state_dict(),
        'loss_function': lossfn,
        'optimizer': optim,}
        torch.save(modeldict ,'./PCM_state.pth.tar')
    num_correct=0
    total = 0

    for i, (images, labels) in enumerate(train_dataLoader):
        outputs = model(images)
        outputs = torch.argmax(outputs, dim=1).float()
        labels = labels.float()
        loss = lossfn(outputs, labels)
        total += labels.size(0)
        num_correct += (outputs == labels).sum()
        loss.requires_grad = True
        optim.zero_grad() #reset the gradients, before starting backpropagation
        loss.backward() #if you first call this, gradients will sum up with the previous ones
        optim.step() #once backpropagation is over => change parameters

    print(f'Epoch: {epoch+1}, Loss: {loss.item()}')
    print('the model got {}/{} an accuracy of {}, on epoch:{}'.format(num_correct,total,(num_correct/total),epoch+1))
    acc.append((num_correct/total)*100)
    print(acc)
def checkValues(loader,network,stage):
    num_correct = 0
    total = 0 
    model.eval()
    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            if i == 100:
                total = 100
                break
                
            outputs = network(images)
            predicted = outputs.data
            print(predicted)
            total += labels.size(0)
            num_correct += (predicted == labels).sum().item()
        print('the model got {}/{} an accuracy of {}, on:{}'.format(num_correct,total,(num_correct/total),stage))
        model.train()

checkValues(train_dataLoader, model,'training')

checkValues(test_dataLoader, model,'testing')

Plot_data(range(epochs),acc,xLabel='Epoca',yLabel='Eficacia',title='Eficacia en entrenamiento')