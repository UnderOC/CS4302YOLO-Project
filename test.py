
import torch

A = torch.arange(16,device='cuda', dtype=torch.float).reshape(4,4)

# B = A*A
B = torch.mm(A,A)

print(A)

print(B)