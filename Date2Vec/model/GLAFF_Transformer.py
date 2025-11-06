'''
# Experiment Objective:
# Test the ability of Primal Position as the position encoding in Transformer.
# Compare D2V, T2V, and position-based encodings.
# Use these position encodings and input them into a Transformer model.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
import os
import matplotlib.pyplot as plt
import numpy as np
from plugin.Plugin.model import Plugin


class PositionalEmbedding(nn.Module):
    """
    Positional encoding using sine and cosine functions, as used in Transformers.
    """

    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()

        pe = torch.zeros(max_len, d_model).float()
        pe.requires_grad = False

        # Generate the position encodings using sine and cosine functions
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Get the positional encodings for the given sequence length
        output = self.pe[:, :x.size(1)]
        return output


class GLAFF_Transformer(nn.Module):
    """
    Transformer model with Primal Position as the positional encoding.
    """

    def __init__(self, configs):
        super(GLAFF_Transformer, self).__init__()

        # Initialize model parameters from configs
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.d_feature = configs.d_feature
        self.d_model = configs.d_model

        # Linear layer to project input features to model dimension
        self.transformer_proj = nn.Linear(self.d_feature, self.d_model)

        # Output projection layer for the prediction
        self.output_proj = nn.Linear(self.seq_len, self.pred_len)

        # Positional embedding (Primal Position)
        self.Primal_position = PositionalEmbedding(self.d_model)

        # Transformer Encoder and Decoder setup
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model, configs.d_ff, dropout=configs.dropout, activation=configs.activation
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=torch.nn.BatchNorm1d(configs.d_model)
        )

        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        FullAttention(True, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model, configs.d_ff, dropout=configs.dropout, activation=configs.activation,
                ) for _ in range(configs.d_layers)
            ],
            norm_layer=torch.nn.BatchNorm1d(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )

        # Initialize the Plugin module
        self.plugin = Plugin(configs, configs.d_feature)

    def forward(self, x_enc, x_mark_enc, y_batch, x_mark_dec, mode):
        """
        Forward pass through the model.

        Args:
            x_enc: Encoded input tensor (B, L, D)
            x_mark_enc: Time features for input (B, L, D)
            y_batch: Target tensor for prediction (not used in this code)
            x_mark_dec: Time features for target (B, O, D)
            mode: Mode of operation ('norm' for normalization, 'denorm' for denormalization)

        Returns:
            pred: The final predicted output tensor
        """
        # Make copies of the input for later use
        x_enc_copy, x_mark_enc_copy, x_mark_dec_copy = x_enc.clone(), x_mark_enc.clone(), x_mark_dec.clone()

        # Prepare input data by projecting to higher dimensions
        x = x_enc
        x = self.transformer_proj(x)

        # Get positional encoding (Primal Position)
        position = self.Primal_position(x)

        # Combine input and position encoding for Transformer input
        Transformer_input = x + position

        # Pass input through Transformer Encoder and Decoder
        encoder_output = self.encoder(Transformer_input)
        decoder_output = self.decoder(Transformer_input, encoder_output)

        # Project the decoder output to the desired prediction length
        output = self.output_proj(decoder_output.permute(0, 2, 1)).permute(0, 2, 1)

        # Pass the output through the Plugin module for final prediction
        pred = self.plugin(x_enc_copy, x_mark_enc_copy, output, x_mark_dec_copy[:, -self.pred_len:, :])

        return pred  # Return the predicted output
