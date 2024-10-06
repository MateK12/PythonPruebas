import cv2
from PIL import Image
from torchvision import transforms as tr
import matplotlib.pyplot as plt

trans = tr.Compose(
   [ tr.RandomVerticalFlip(0.2),
    tr.RandomHorizontalFlip(0.4)]
)

fig, canva = plt.subplots()

def change_image(path,iter,transforms):
    img = Image.open(path)
    imgs =[]
    for i in range(iter):
        img = transforms(img)
        imgs.append(img)
    for pics in enumerate(imgs):
        print(pics)
        plt.imshow(pics)
        plt.show()

change_image('./picture.png',3,trans)