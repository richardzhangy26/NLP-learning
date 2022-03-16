# Bert 论文代码解读

 [BERT_2018.pdf](BERT_2018.pdf) 

[Bert 网页](https://arxiv.org/pdf/1810.04805.pdf)

[李沐](https://www.bilibili.com/video/BV1PL411M7eQ?spm_id_from=333.999.0.0)  

<img src="https://cdn.jsdelivr.net/gh/richardzhangy26/Pic/src/image-20220315153155250.png" alt="image-20220315153155250" style="zoom:50%;" />

**预训练好的bert只需要微调再加一个输出层来做到不同的自然语言任务**

<img src="https://cdn.jsdelivr.net/gh/richardzhangy26/Pic/src/image-20220315155613969.png" alt="image-20220315155613969" style="zoom:50%;" />

Bert的复杂度跟层数是线性关系，跟宽度是平方关系

## 超参数的数量由来：

![image-20220315160705059](https://cdn.jsdelivr.net/gh/richardzhangy26/Pic/src/image-20220315160705059.png)

30K *H 是指词表总数乘以隐藏层

4*H^2是QKV和投影层的参数

8*H^2是MLP映射

![image-20220315165732004](https://cdn.jsdelivr.net/gh/richardzhangy26/Pic/src/image-20220315165732004.png)

bert的编码器可以是两个句子的组合
