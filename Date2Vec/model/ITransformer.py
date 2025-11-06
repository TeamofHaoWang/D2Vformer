import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.iTransformer_EncDec import Encoder, EncoderLayer
from layers.iTransformer_SelfAtt_Famliy import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
# from plugin.Plugin.model import Plugin


class iTransformer(nn.Module):
    """
    iTransformer model as described in the paper: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, configs):
        super(iTransformer, self).__init__()

        # Initialize model parameters
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = False
        self.use_norm = True

        # Embedding layer for input features and time-series data
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)

        # Encoder architecture with attention layers
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        # Projection layer for final output
        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        Forecasting step for time-series data.

        Args:
            x_enc (torch.Tensor): Encoded input sequence
            x_mark_enc (torch.Tensor): Encoded time features for input
            x_dec (torch.Tensor): Decoding input sequence (unused in this model)
            x_mark_dec (torch.Tensor): Decoding time features (unused in this model)

        Returns:
            dec_out (torch.Tensor): The predicted output tensor
        """
        if self.use_norm:
            # Normalize input data (from Non-stationary Transformer approach)
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        _, _, N = x_enc.shape  # B: batch size, L: sequence length, N: number of features

        # Input embedding (transforming time series and covariates into higher dimensions)
        enc_out = self.enc_embedding(x_enc, x_mark_enc)

        # Process with encoder (attention layers, layer normalization, and feed-forward)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Project encoder output to prediction space
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N]  # Filter covariates

        if self.use_norm:
            # De-normalize the output (from Non-stationary Transformer)
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mode):
        """
        Forward pass for the model.

        Args:
            x_enc (torch.Tensor): Encoded input tensor of shape [B, L, D]
            x_mark_enc (torch.Tensor): Time features for input
            x_dec (torch.Tensor): Decoding input (unused in this model)
            x_mark_dec (torch.Tensor): Time features for decoding (unused in this model)
            mode (str): Mode for operation ('train' or 'test')

        Returns:
            pred (torch.Tensor): The final prediction tensor of shape [B, L, D]
        """
        # Perform forecasting based on encoder input
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)

        # Extract the prediction for the required time window
        pred = dec_out[:, -self.pred_len:, :]
        return pred  # [B, L, D]
