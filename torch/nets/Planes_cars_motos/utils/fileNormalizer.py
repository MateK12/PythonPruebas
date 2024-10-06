import os
import csv
from PIL import Image
def normalize_files(path:str,main_name:str):

    for i,file_name in enumerate(os.listdir(path)):
        filePath = "{}{}".format(path,file_name)
        extension = file_name.split('.')[1]
        new_name = "{}{}_{}.{}".format(path,main_name,i+1,extension)
        os.rename(filePath,new_name)
# normalize_files('./assets/planes/','plane')

def rgb_normalizer(path):
    for i, file in enumerate(os.listdir(path)):
        Image.open(os.path.join(path,file))
        



