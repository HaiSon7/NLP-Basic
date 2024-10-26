import torch
from torch import nn
from multi_head_attention import MultiHeadAttention
from feed_forward import FeedForward

class Encoder(nn.Module):
    def __init__(self,expansion_factor = 4,d_model = 300,num_heads = 4,drop_out = 0.1):
        super(Encoder,self).__init__()
        self.matn = MultiHeadAttention(d_model,num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.feedforward = FeedForward(d_model,d_model*expansion_factor)
        self.dropout = nn.Dropout(drop_out)

    def forward(self,x,mask= None):
        matn_output = self.matn(x,x,x,mask)
        x = self.norm1(x+self.dropout(matn_output))
        ffn_output = self.feedforward(x)
        x = self.norm2(x+self.dropout(ffn_output))

        return x



