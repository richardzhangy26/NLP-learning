from cv2 import GEMM_2_T
import numpy as np
import pandas as pd
import pickle
import jieba
import os
from torch import embedding
from tqdm import tqdm

def load_stop_words(files=r'word_2_vec-main\stopwords.txt'):
    with open(files,'r',encoding='utf-8') as f:
        return f.read().split('\n')
def cut_words(files=r"word_2_vec-main\数学原始数据.csv"):
    stop_words = load_stop_words()
    all_data = pd.read_csv(files,encoding='gbk',names=['data'])['data']
    results = []
    for words in all_data:
        c_words  = jieba.lcut(words)
        results.append([word for word in c_words if word not in stop_words])
    return results

def get_dict(data):
    index_2_word = []
    for words in data:
        for word in words:
            if word not in index_2_word:
                index_2_word.append(word)
    word_2_index = {word:index for index,word in enumerate(index_2_word)}
    word_size = len(word_2_index)
    word_2_onehot = {}
    for word,index in word_2_index.items():
        one_hot = np.zeros((1,word_size))
        one_hot[0,index] = 1
        word_2_onehot[word] = one_hot
    return word_2_index, index_2_word,word_2_onehot

def softmax(x):
    ex = np.exp(x)
    return ex/np.sum(ex,axis=1,keepdims=True)
if __name__ =="__main__":
    data = cut_words()
    word_2_index, index_2_word,word_2_onehot  = get_dict(data)
    word_size =  len(word_2_index)
    embedding_dim = 100
    lr  = 0.01
    epoch = 10
    n_gram = 3

    w1  = np.random.normal(-1,1,size=(word_size,embedding_dim))
    w2 = np.random.normal(-1,1,size = (embedding_dim,word_size))

    for e in range(epoch):
        for words in tqdm(data):
            for n_index, num_word in enumerate(words):
                now_word_onehot = word_2_onehot[num_word]
                other_words = words[max(n_index-n_gram,0):n_index] + words[n_index+1:n_index+n_gram]
                
                for other_word in other_words:
                    other_word_onehot = word_2_onehot[other_word]
                    hidden = now_word_onehot @ w1 #shape[1,100]        
                    p = hidden @ w2 #shape[1,word_size]
                    pre = softmax(p)
                    # loss = -np.sum(other_word_onehot *np.log(pre))
                    #  A @ B =C
                    # delta_C=G
                    # delta_A = G @ B.T
                    # delta_B = A.T @ G
                    # https://zhuanlan.zhihu.com/p/25723112(softmax求导)
                    G2 = pre - other_word_onehot #softmax 求导即 pre-onehot
                    delta_w2 = hidden.T @ G2
                    G1 = G2 @ w2.T
                    delta_w1 = now_word_onehot.T @ G1

                    w1-= lr * delta_w1
                    w2-= lr * delta_w2
    with open(r"word2vec.pkl",'wb')as f:
        pickle.dump([w1,word_2_index,index_2_word],f)