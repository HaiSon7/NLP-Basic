from torch import nn
import torch
from PositionalEmbedding import PositionalEmbedding
from EncoderLayer import Encoder
from DecoderLayer import Decoder
from Genarate_Mask import generate_mask
class Transformer(nn.Module):
    def __init__(self,src_vocab_size, tgt_vocab_size,d_model,num_heads,num_layers,
                 expansion_factor,seq_len,drop_out):
        super(Transformer,self).__init__()
        self.enc_embed = nn.Embedding(src_vocab_size, d_model)
        self.dec_embed = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_embed = PositionalEmbedding(seq_len,d_model)
        self.encoder = nn.ModuleList([Encoder(expansion_factor,d_model,num_heads,drop_out) for _ in range(num_layers)])
        self.decoder = nn.ModuleList([Decoder(expansion_factor,d_model,num_heads,drop_out) for _ in range(num_layers)])

        self.fc = nn.Linear(d_model,tgt_vocab_size)
        self.drop_out = drop_out
        self.generate_mask = generate_mask(src_vocab_size,tgt_vocab_size)

    def forward(self,src,tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))

        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        output = self.fc(dec_output)
        return output

