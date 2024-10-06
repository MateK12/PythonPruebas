import pandas as pd
from torch.utils.data import DataLoader, dataset, random_split
from utils.dataset import BreastCancerDataset 
from random import randint
import torch.nn as nn
import torch
main_dataset = BreastCancerDataset('./assets/Breast_cancer_data.csv')
train_dataset,test_dataset = random_split(main_dataset,[505,64])



num_epochs = 100
learning_rate = 0.007
batch_size = 16

train_dataLoader = DataLoader(train_dataset,shuffle=True,batch_size=batch_size)
test_dataLoader = DataLoader(test_dataset,shuffle=True,batch_size=batch_size)


class Breast_cancer_network(nn.Module):
    def __init__(self):
        super(Breast_cancer_network, self).__init__()
        self.input = nn.Linear(5,30)
        self.fc1 = nn.Linear(30, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, 1)
        self.sigm = nn.Sigmoid()
    def forward(self, x):
        x = self.input(x)
        x = self.relu(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x =self.sigm(x)
        return x
model = Breast_cancer_network()

lossFN = nn.BCELoss()
optimizer = torch.optim.Adam(params=model.parameters(),lr=learning_rate)

model.train()
for epoch in range(num_epochs):
    if epoch == num_epochs-1:
        dict = {
            'state_dict':model.state_dict(),
            'loss_function': lossFN,
            'optimizer': optimizer,
            'epoch': epoch+1,
        }
        torch.save(dict,'./assets/breast_cancer_trained_model.pth.tar')
    for i, (features , y) in enumerate(train_dataLoader):
        features = features.float()
        labels = y.float()
        predictions = model(features)
        predictions = predictions.squeeze()
        loss = lossFN(predictions, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("Epoch [{}/{}], Loss: {:.4f}".format(epoch+1, num_epochs, loss.item()))

num_corrects=0
total = 0
model.eval()
with torch.no_grad():
    for x,tar in test_dataLoader:
        out = model(x.float())
        total += tar.size(0)
        for i,o in enumerate(out):
            print(o)
            print(i)
            if o>0.5:
                out[i] =1
            else:
                out[i] = 0
        out = out.squeeze()
        print(out.shape)
        print(tar.shape)
        tar = tar.float()
        total_sum = (out==tar).sum() #returns the sum of all the elements, trues=1 falses=0
        num_corrects+= total_sum.item() #returns a single scalar (only for 1dimensional tensors)
    print('The model got {}/{} an accuracy of {}%'.format(num_corrects,total,(num_corrects/total)*100))



