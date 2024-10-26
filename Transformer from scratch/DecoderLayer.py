import torch
from torch import nn
from multi_head_attention import MultiHeadAttention
from feed_forward import FeedForward
class Decoder(nn.Module):
    def __init__(self,d_model,expansion_factor = 4,num_heads = 4,drop_out = 0.1):
        super(Decoder,self).__init__()
        self.matn = MultiHeadAttention(d_model,num_heads)
        self.cross_attn = MultiHeadAttention(d_model,num_heads)
        self.feed_forward = FeedForward(d_model,d_model*expansion_factor)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.drop_out = nn.Dropout(drop_out)

    def forward(self,x,enc_output,src_mask,tgt_mask):
        matn = self.matn(x,x,x,tgt_mask)
        x = self.norm1(x+self.drop_out(matn))
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x


