
import math
from typing import Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class TransformerModel(nn.Module):

    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
        
        self.d_model = d_model
        
        self.embedding_encoder = nn.Embedding(ntoken, d_model) # d_model: embed dimension, e.g. 200
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # d_model: input feature dimension, i.e. embedding layer dimension 
        # d_hid: fully connected layer dimension
        # often set d_model == d_hid
        # Note: d_hid won't impact the output dimension of TransformerEncoderLayer
        # the output dimension is still d_model 
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        
        self.decoder = nn.Linear(d_model, ntoken)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding_encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]
            
            ex:
            bptt = 35
            batch_size = 20
            src.shape: torch.Size([35, 20])

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        # import pdb; pdb.set_trace()
        
        # after encoder, src shape change from torch.Size([35, 20]) -> torch.Size([35, 20, 200])
        # input: [seq_len, batch_size]
        # output: [seq_len, batch_size, d_model(embedding dimension)]
        src = self.embedding_encoder(src) * math.sqrt(self.d_model) 
        
        # input/output dimension not changed
        src = self.pos_encoder(src)  
        
        # input src:  [seq_len, batch_size, d_model]
        # input mask: [seq_len, seq_len]
        # output: [seq_len, batch_size, d_model]
        output = self.transformer_encoder(src, src_mask) # output.shape: torch.Size([35, 20, 200])
        
        # input: [seq_len, batch_size, d_model]
        # ouptut: [seq_len, batch_size, vocab_size]
        output = self.decoder(output) # output.shape: torch.Size([35, 20, 28782])
        return output


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
    

def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

