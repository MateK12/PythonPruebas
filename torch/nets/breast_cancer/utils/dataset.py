from torch.utils.data.dataset import Dataset
import pandas as pd
import torch
class BreastCancerDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):

        x = self.data.iloc[index]
        y = x.pop('diagnosis')
        x = torch.tensor(x.values)
        return (x,y)