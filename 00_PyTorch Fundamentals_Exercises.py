import torch 
import numpy as np


# 1. create a random tensor with shape (7,7)

tensor = torch.rand(7,7)

# 2. create a second random tensors with shape (1,7) and matmul them together

tensor_2 = torch.rand(1,7)	

tensor_3 = torch.mm(tensor,tensor_2.T)

# or 

tensor_3 = tensor @ tensor_2.T

# 3. set random seed to 0 and redo  1 and 2. Do you get the same result?
torch.manual_seed(0)
tensor = torch.rand(7,7)
tensor_2 = torch.rand(1,7)
tensor_3 = torch.matmul(tensor,tensor_2.T)

# set random seed to 0 and redo experiments on GPU
torch.manual_seed(0)
tensor = torch.rand(7,7,device='cuda')
tensor_2 = torch.rand(1,7, device='cuda')   
tensor_3 = torch.matmul(tensor,tensor_2.T)

# find maximum and minimum values of tensor_3
max_value = torch.max(tensor_3)
min_value = torch.min(tensor_3)


# 10. make a random tensor of size (1,1,1,10) and then remove the dimensions of size 1
high_dim_tensor = torch.rand(1,1,1,10)
print(high_dim_tensor.shape)
high_dim_tensor = torch.squeeze(high_dim_tensor)
print(high_dim_tensor.shape)


