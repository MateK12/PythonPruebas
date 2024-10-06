from torch.utils.data.dataset import Dataset
import pandas as pd
import os
from skimage import io
import torch 
from PIL import Image

class planes_cars_motos_Dataset(Dataset):
    def __init__(self,file, root,transform=None):
        self.annotations = pd.read_csv(file)
        self.root = root
        self.transform = transform
    def __len__(self): # with __dunder__ methods its possible to modify native behaviour of built in functions
        return len(self.annotations)#whenever I call the len() method over an instance of this class, it will return this

    def __getitem__(self,index):
    
        img_path = os.path.join(self.root,self.annotations.iloc[index,0]) 
        x = io.imread(img_path) #reads the image (all the class 'io' is for saving, uploading and creating images)
        x = Image.open(img_path)
        x = x.convert('RGB')
        if self.transform: #if it has a transform such as Totensor(), it applies it
            x = self.transform(x)

        y = torch.tensor(int(self.annotations.iloc[index,1])) #converts into a tensor the data found at row=index and column=1 (targets)

        return (x,y)
    
