'''
# Experiment Objective:
# Test the ability of Primal Position as the positional encoding in Transformer.
# Compare D2V, T2V, and position-based encodings.
# Use these position encodings and input them into a Transformer model.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer
from layers.SelfAttention_Family import M_FullAttention, AttentionLayer
import matplotlib.pyplot as plt
import numpy as np
import math


class PositionalEmbedding(nn.Module):
    """
    Positional encoding using sine and cosine functions as in Transformer models.
    """
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()

        # Initialize positional encoding matrix
        pe = torch.zeros(max_len, d_model).float()
        pe.requires_grad = False

        # Create position vector and apply sine and cosine functions
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Return the positional encoding for the given sequence length
        output = self.pe[:, :x.size(1)]
        return output


class Position_Transformer(nn.Module):
    """
    Primal Position as positional encoding in Transformer architecture.
    """
    def __init__(self, configs):
        super(Position_Transformer, self).__init__()

        # Initialize model parameters
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.d_feature = configs.d_feature
        self.d_model = configs.d_model

        # Linear layer to project input features to model dimension
        self.transformer_proj = nn.Linear(self.d_feature, self.d_model)

        # Output projection layer
        self.output_proj = nn.Linear(self.seq_len, self.pred_len)

        # Positional encoding (Primal Position)
        self.Primal_position = PositionalEmbedding(self.d_model)

        # Transformer Encoder setup
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        M_FullAttention(configs.factor, attention_dropout=configs.dropout,
                                        output_attention=False), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=torch.nn.BatchNorm1d(configs.d_model)
        )

        # Transformer Decoder setup
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        M_FullAttention(configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    AttentionLayer(
                        M_FullAttention(configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for _ in range(configs.d_layers)
            ],
            norm_layer=torch.nn.BatchNorm1d(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )

    def forward(self, x_enc, x_mark_enc, y_batch, x_mark_dec, mode):
        """
        Forward pass through the model.

        Args:
            x_enc (torch.Tensor): Encoded input sequence [B, L, D]
            x_mark_enc (torch.Tensor): Time features for input [B, L, D]
            y_batch (torch.Tensor): Target tensor (unused in this model)
            x_mark_dec (torch.Tensor): Time features for target (unused in this model)
            mode (str): Mode of operation ('train' or 'test')

        Returns:
            output (torch.Tensor): Final output after decoding [B, L, D]
        """
        # Project input to model dimension
        B, L, D = x_enc.shape
        x = x_enc
        x = self.transformer_proj(x)

        # Get Primal Position (positional encoding)
        position = self.Primal_position(x)

        # Combine input features with positional encoding
        Transformer_input = x + position

        # Pass through Transformer Encoder and Decoder
        encoder_output = self.encoder(Transformer_input)
        decoder_output = self.decoder(Transformer_input, encoder_output)

        # Project the decoder output to prediction space
        output = self.output_proj(decoder_output.permute(0, 2, 1)).permute(0, 2, 1)

        return output  # Return the final output [B, L, D]
