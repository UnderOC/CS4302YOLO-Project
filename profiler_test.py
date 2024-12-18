import torch

x = torch.randn((1, 1), requires_grad=True)
with torch.autograd.profiler.profile(record_shapes=True) as prof:
    for _ in range(100):
        y = x ** 2
        y.backward()

# 获取 key_averages 对象
averages = prof.key_averages()
print(averages)

# 打印表格，不进行排序
print(type(averages.table()))
