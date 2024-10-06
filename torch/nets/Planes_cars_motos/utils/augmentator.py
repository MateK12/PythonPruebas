import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision.utils import save_image
import os
from PIL import Image

all_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.ToPILImage(),
    transforms.Grayscale(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(45),
    transforms.ColorJitter(brightness=0.5),
    transforms.ToTensor(),
    transforms.Resize((224,224)),
]) 

def Augmentator(images,custom_transforms,item,destiny_path,limit=2000,multiplyer=10):
    img_num=0
    
    for i in range(multiplyer):
        for i, img in enumerate(os.listdir(images)):
            image = Image.open('{}/{}'.format(images,img))
            transformed_image = (custom_transforms(image))
            print('{}_{} created'.format(item,img_num))
            img_num+=1
            save_image(transformed_image,'{}/{}_num{}.png'.format(destiny_path, item,img_num))
            if img_num == limit:
                print('{} images were created'.format(img_num))
                exit()
    print('{} images were created'.format(img_num))

    
           

Augmentator('../assets/cars',all_transforms,'car','./augmentator/augmented_images/car')