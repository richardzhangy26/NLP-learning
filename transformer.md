# 多头注意力机制

![image-20220218162310399](C:\Users\86173\AppData\Roaming\Typora\typora-user-images\image-20220218162310399.png)

实际代码使用矩阵，方便并行

1. 绿色一行代表一个词的向量
2. dk表示k的维度，除以根号dk是为了降低softmax的值，方便梯度下降

![](C:\Users\86173\AppData\Roaming\Typora\typora-user-images\image-20220218162925410.png)

![image-20220218163011792](C:\Users\86173\AppData\Roaming\Typora\typora-user-images\image-20220218163011792.png)

# 残差和layNorm

[b站DASOU](https://www.bilibili.com/video/BV1Di4y1c7Zm?p=4&spm_id_from=pageDriver)



# 代码分析

## 整体结构



<img src="C:\Users\86173\AppData\Roaming\Typora\typora-user-images\image-20220220183929908.png" alt="image-20220220183929908"  />

1. 从整体网路结构来看，分为三个部分：编码层，解码层，输出层

   ```
   ## 1. 从整体网路结构来看，分为三个部分：编码层，解码层，输出层
   class Transformer(nn.Module):
       def __init__(self):
           super(Transformer, self).__init__()
           self.encoder = Encoder()  ## 编码层
           self.decoder = Decoder()  ## 解码层
           self.projection = nn.Linear(d_model, tgt_vocab_size, bias=False) 
           ## 输出层 d_model 是我们解码层每个token输出的维度大小，之后会做一个 	tgt_vocab_size 大小的softmax
       def forward(self, enc_inputs, dec_inputs):
           ## 这里有两个数据进行输入，一个是enc_inputs 形状为[batch_size, src_len]，主要是作为编码段的输入，一个dec_inputs，形状为[batch_size, tgt_len]，主要是作为解码端的输入
           ## enc_inputs作为输入 形状为[batch_size, src_len]，输出由自己的函数内部指定，想要什么指定输出什么，可以是全部tokens的输出，可以是特定每一层的输出；也可以是中间某些参数的输出；
           ## enc_outputs就是主要的输出，enc_self_attns这里没记错的是QK转置相乘之后softmax之后的矩阵值，代表的是每个单词和其他单词相关性；
           enc_outputs, enc_self_attns = self.encoder(enc_inputs)
   
           ## dec_outputs 是decoder主要输出，用于后续的linear映射； dec_self_attns类比于enc_self_attns 是查看每个单词对decoder中输入的其余单词的相关性；dec_enc_attns是decoder中每个单词对encoder中每个单词的相关性；
           dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_inputs, enc_outputs)
   
           ## dec_outputs做映射到词表大小
           dec_logits = self.projection(dec_outputs) # dec_logits : [batch_size x src_vocab_size x tgt_vocab_size]
           return dec_logits.view(-1, dec_logits.size(-1)), enc_self_attns, dec_self_attns, dec_enc_attns
   ```

## 模型参数

```
d_model = 512  # Embedding Size
d_ff = 2048  # FeedForward dimension
d_k = d_v = 64  # dimension of K(=Q), V
n_layers = 6  # number of Encoder of Decoder Layer
n_heads = 8  # number of heads in Multi-Head Attention
```



nn.embedding(词表的大小，词的维度) 
