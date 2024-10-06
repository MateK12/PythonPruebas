import torch
import numpy as np
from torch import nn
with open('dataset/shekpeare.txt','r',encoding='utf-8' ) as t:
    text = t.read()


chars = sorted(list(set(text)))
vocab_size = len(chars)

back = { ch:i for i,ch in enumerate(chars)}
kcab = { i:ch for i,ch in enumerate(chars)}
encode = lambda x: [back[c]for c in x]
decode = lambda x: ''.join([kcab[i] for i in x])

data = torch.tensor(encode(text), dtype=torch.long)

n = int(0.9* len(data))

train_data = data[:n]
validation_data= data[:n]
rand_list = np.random.randint(0,64,size=100)

batch_size = 4
block_size = 8
def get_batch(split):
    data = train_data if split == 'train' else validation_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

x_batch, y_batch = get_batch('train')
print(x_batch)
# print(data.shape)
# print(data.dtype)
# print(data[:1000])

class BigramModelLanguege(nn.Module):
    def __init__(self, vocab_size):
        super(BigramModelLanguege, self).__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
    def forward(self,x,y):
        logits = self.token_embedding_table(x)
        return logits
