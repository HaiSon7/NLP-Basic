import torch
import math
from torch import nn
class MultiHeadAttention(nn.Module):
    def __init__(self,d_model,num_heads):
        super(MultiHeadAttention, self).__init__()

        self.d_k = d_model // num_heads
        self.num_heads = num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self,Q, K, V, mask=None):
        dk = Q.size(1)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(dk)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = torch.softmax(scores, dim=-1)

        context = torch.matmul(attn, V)

        return context
    def split_heads(self,x):
        batch_size = x.size(0)
        seq_len = x.size(1)
        x = x.view(batch_size,seq_len,self.num_heads,self.d_k)
        return x



    def forward(self,Query,Key,Value, mask=None):

        Q = self.W_q(Query)
        K = self.W_k(Key)
        V = self.W_k(Value)


        q = self.split_heads(Q)
        k = self.split_heads(K)
        v = self.split_heads(V)

        attention_weights =  self.scaled_dot_product_attention(q,k,v,mask)


        batch_size = k.size(0)
        seq_len = k.size(1)

        output = attention_weights.contiguous().view(batch_size,seq_len,self.d_k *self.num_heads)



        return self.W_o(output)

batch_size = 32
seq_len = 25

d_model = 512
num_heads = 8
multihead_attention = MultiHeadAttention(d_model, num_heads)

#(batch_size, seq_len, d_model)
input_tensor = torch.rand(batch_size,seq_len,d_model)
print(multihead_attention.forward(input_tensor,input_tensor,input_tensor).shape)


    

