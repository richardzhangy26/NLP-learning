import torch
x = torch.zeros(2,3)
print(x.unsqueeze(0).transpose(1,0).size())
