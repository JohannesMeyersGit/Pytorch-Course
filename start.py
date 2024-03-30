import torch
import pandas as pd
import numpy as np  # we're going to be creating arrays to feed into our model
import matplotlib.pyplot as plt  # for visualization

print(torch.cuda.is_available())
# list available cuda devices  

print(torch.cuda.device_count())
torch.cuda.get_device_name(0)
dev = torch.cuda.current_device() 
print(torch.cuda.get_device_properties(dev))

## Introduction to Tensor

# creating tensors
scalar = torch.tensor(7)
print(scalar.ndim) # 0 dimension
print(scalar.shape) # scalar has no shape because it's a scalar
item = scalar.item()

# vector

vector = torch.tensor([1, 2, 3, 4]) # 1 dimension vector
print(vector.ndim)
print(vector.shape)

# matrix
matrix = torch.tensor([[1, 2], [3, 4]]) # 2 dimension matrix
print(matrix.ndim)
print(matrix.shape)

# 3 dimension tensor
tensor_3d = torch.tensor([
    [[1, 2], [3, 4]],
    [[5, 6], [7, 8]]
])  # 3 dimension tensor
print(tensor_3d.ndim)
print(tensor_3d.shape)

float_tensor = torch.tensor([1, 2, 3, 4], dtype=torch.float32, requires_grad=False, device='cpu')

print(float_tensor.dtype)  

float_16_tensor = float_tensor.type(torch.float16)  
print(float_16_tensor.dtype)
print(float_16_tensor.device)
print(float_16_tensor.requires_grad)
float_16_tensor = float_16_tensor.to('cuda')
float_16_tensor.requires_grad = True
print(float_16_tensor.dtype)
print(float_16_tensor.device)
print(float_16_tensor.requires_grad)

x = torch.randn(2,3,4)

print(x.shape)

# switch the first and second dimension
x = torch.permute(x,(1,0,2))

print(x.shape)

# write me a matrix with three dimnesions, where the first dimension has 2 elements, the second dimension has 3 elements, and the third dimension has 4 elements.
x = torch.randn(2,3,4) 


a = np.arange(15).reshape(3, 5)

tensor = torch.from_numpy(a)

print(tensor.dtype)

# change dtype to float32
tensor = tensor.type(torch.float32)

print(tensor.dtype)

# change tensor to numpy array
numpy_arr = tensor.numpy()
print(numpy_arr.dtype)

## reproducibility of random numbers in pytorch
ran = torch.rand(2,3)

# set seed 
torch.manual_seed(42)
ran2 = torch.rand(2,3) # same random numbers
torch.manual_seed(42) # reset seed to 42
ran3 = torch.rand(2,3) # same random numbers

print(ran, ran2, ran3, sep='\n')    


## Running tensors on GPU
# check if cuda is available
print(torch.cuda.is_available()) # should be True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device.type)

# generate a tensor on cpu
tensor_cpu = torch.rand(2,3,dtype=torch.float32, requires_grad=True, device='cpu')
tensor_gpu = torch.rand(2,3,dtype=torch.float32, requires_grad=True, device='cuda')

# move tensor from cpu to gpu 
tensor_cpu_to_gpu = tensor_cpu.to(device)

# move tensor from gpu to cpu
tensor_gpu_to_cpu = tensor_gpu.to('cpu')


