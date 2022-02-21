from math import sqrt
import torch
import torch.nn as nn


class Self_Attention(nn.Module):
    # input : batch_size * seq_len * input_dim
    # q : batch_size * input_dim * dim_k
    # k : batch_size * input_dim * dim_k
    # v : batch_size * input_dim * dim_v
    def __init__(self, input_dim, dim_k, dim_v):
        super(Self_Attention, self).__init__()
        self.q = nn.Linear(input_dim, dim_k)
        self.k = nn.Linear(input_dim, dim_k)
        self.v = nn.Linear(input_dim, dim_v)
        self._norm_fact = 1 / sqrt(dim_k)

    def forward(self, x):
        Q = self.q(x)  # Q: batch_size * seq_len * dim_k
        K = self.k(x)  # K: batch_size * seq_len * dim_k
        V = self.v(x)  # V: batch_size * seq_len * dim_v

        atten = nn.Softmax(dim=-1)(torch.bmm(Q, K.permute(0, 2, 1))) * self._norm_fact  # Q * K.T() # batch_size * seq_len * seq_len
        # bmm 对同一batch矩阵对应相乘。 permute 改变张量的维度
        output = torch.bmm(atten, V)  # Q * K.T() * V # batch_size * seq_len * dim_v

        return output
X = torch.randn(4,3,2)
print(X)
print(X.size())
self_attention = Self_Attention(2,4,5)#q(2,4),k(2,4),v(2,5)
res = self_attention(X)
print(res)
print(res.size())