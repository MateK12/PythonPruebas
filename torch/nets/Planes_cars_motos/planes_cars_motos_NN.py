import torch.nn as nn
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, random_split
from utils.custom_dataset import planes_cars_motos_Dataset
import numpy as np
import math
from utils.trunc import redondear_si_mayor_65



batch_size = 32
learning_rate = 0.0004
epochs = 30


alltransforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224,224)),
])

dataset = planes_cars_motos_Dataset('/home/mateo/Escritorio/Python/torch/nets/Planes_cars_motos/assets/planes_cars_motos.csv','/home/mateo/Escritorio/Python/torch/nets/Planes_cars_motos/assets/images',transform=alltransforms)
print(len(dataset))
train_dataset, test_dataset = random_split(dataset,[678,75])

train_dataLoader = DataLoader(train_dataset, batch_size=batch_size,shuffle=True)
test_dataLoader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

print('dataloaders and datasets loaded successfully')

class Plane_cars_motos_model(nn.Module):
    def __init__(self):
        super(Plane_cars_motos_model,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,kernel_size=3,stride=2,out_channels=122)
        self.conv2 = nn.Conv2d(in_channels=122,kernel_size=3,stride=2,out_channels=64)
        self.maxPool_layer1 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv3 = nn.Conv2d(in_channels=64,kernel_size=3,stride=2,out_channels=32)
        self.maxPool_layer2 = nn.MaxPool2d(kernel_size=2,stride=2)

        self.dense_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1152,32),
            nn.ReLU(),
            nn.Linear(32,3)
        )

    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.maxPool_layer1(x)
        x = self.conv3(x)
        x = self.maxPool_layer2(x)
        x = self.dense_layer(x)
        x = torch.max(x)
        return x
    

model = Plane_cars_motos_model()
lossfn = nn.L1Loss()
optim = torch.optim.SGD(model.parameters(), lr=learning_rate,momentum=0.8, weight_decay=0.005)

for epoch in range(epochs):
    if epoch ==29:
        modeldict = {
        'state_dict': model.state_dict(),
        'loss_function': lossfn,
        'optimizer': optim,}
        torch.save(modeldict ,'./PCM_state.pth.tar')

    for i, (images, labels) in enumerate(train_dataLoader):
        optim.zero_grad()
        outputs = model(images)
        loss = lossfn(outputs, labels)
        loss.backward()
        optim.step()
    print(f'Epoch: {epoch+1}, Loss: {loss.item()}')

def checkValues(loader,network,stage):
    num_correct = 0
    total = 0 
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            outputs = network(images)
            predicted= math.trunc(outputs.data)
            print(predicted)
            total += labels.size(0)
            num_correct += (predicted == labels).sum().item()
        print('the model got {}/{} an accuracy of {}, on:{}'.format(num_correct,total,(num_correct/total),stage))
        model.train()

checkValues(train_dataLoader, model,'training')

checkValues(test_dataLoader, model,'testing')