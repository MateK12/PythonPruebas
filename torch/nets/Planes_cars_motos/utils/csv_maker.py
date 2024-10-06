import csv
import os

def csv_maker(destiny_path:str, fields):
    with open(destiny_path,'w', newline='') as file: 
        writer = csv.writer(file)

        writer.writerow(fields)
        for _,car in enumerate(os.listdir('../assets/cars')):
            writer.writerow([car,0])
        for _, plane in enumerate(os.listdir('../assets/planes')):
            writer.writerow([plane,1])
        for _, moto in enumerate(os.listdir('../assets/moto')):
            writer.writerow([moto,2])
        print('csv created successfully')
        print('length of the csv:{}'.format(len(writer)))

# csv_maker('../assets/custom.csv',['feature','target'])#car=0,plane=1,moto=2s


def test():
    with open('../assets/planes_cars_motos.csv','w',newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(['feature','target'])

        for i,img in enumerate(os.listdir('../assets/images_augmented')):
            print('{} of the csv created'.format((i/len(os.listdir('../assets/images_augmented')))*100))
            t = img.split('_')[0]
            if t == 'car':
                csv_writer.writerow([img,0])
            if t =='moto':
                csv_writer.writerow([img,2])
            if t == 'plane':
                csv_writer.writerow([img,1])
test()
def single_Csv(path_to_csv,dir_to_list,item,target:int):
    with open(path_to_csv,'w',newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(['feature','target'])

        for i,img in enumerate(os.listdir(dir_to_list)):
            csv_writer.writerow(['{}_{}'.format(item,i),target])

# single_Csv('./augmentator/csvs/carCSV.csv','../assets/cars','car',0)