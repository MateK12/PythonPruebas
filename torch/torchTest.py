import torch
import numpy as np


device = 'cuda' if torch.cuda.is_available() else 'cpu'
myTensor = torch.tensor([[1,2,3],[4,5,6],[7,8,9]],device=device) #initialize a tensor you can choose the device
print(myTensor)

randTensor = torch.rand(size=(3,3),dtype=torch.float32) #random tensor
print(randTensor)

print(torch.eye(3)) #identity matrix


tensor = torch.arange(1,10,2)
print(tensor.float()) #convert to float
print(tensor.long()) #convert to Int64
print(tensor.half()) #convert to int32

numpy_arr = np.zeros(shape=(3,3))
nd_to_tensor = torch.from_numpy(numpy_arr) #convert np array to torch tensor 
tensor_to_ndarray = nd_to_tensor.numpy() #conver to ndarray (using integrated method)

#broadcasting
tensor1 = torch.rand(size=(5,5))
tensor2 = torch.rand(size=(1,5))
print(tensor2 - tensor1)

teq1 = torch.tensor([1,2,8]) 
teq2 = torch.tensor([1,1,5])

print(torch.eq(teq1, teq2))#compares each element with its respective item with the other vector, returns an array of booleans


x = torch.arange(10)

print(x[(x>3)])
new_tensor = torch.zeros(10,)
print(x.where(x>2,x)) #apply conditions as ternaries, gt2?set_value:value**2

print(new_tensor)
print(torch.tensor([3,2,2,2,4,8,3,9,1,8]).unique()) # returns a tensor with unique values
print(x.numel())#counts the number of elements in the tensor

nineVec = torch.arange(9)
print(nineVec.reshape(3,3)) #reshape tensor