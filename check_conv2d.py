import torch

# torch.nn.functional.silu(torch.randn(1, 1, 2, 2))
# torch.nn.functional.conv2d

A = torch.ones(1, 3, 16, 16)
B = torch.ones(2, 3, 5, 5)

C = torch.nn.functional.conv2d(A, B, stride=(1, 1), padding=(2, 2))
print(C)
print(C.shape)