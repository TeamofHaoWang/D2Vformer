import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer
from layers.SelfAttention_Family import AttentionLayer, FullAttention
from layers.Embed import DataEmbedding


class Transformer(nn.Module):
    """
    Original Transformer model.
    """

    def __init__(self, configs):
        super(Transformer, self).__init__()

        # Model configuration
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.label_len = configs.label_len
        self.d_mark = configs.d_mark
        self.output_attention = False
        self.d_feature = configs.d_feature
        self.d_model = configs.d_model

        # Data embedding (for input and target)
        self.enc_embedding = DataEmbedding(configs.data_dim, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.dec_embedding = DataEmbedding(configs.data_dim, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)

        # Encoder: Stack of EncoderLayers
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=torch.nn.BatchNorm1d(configs.d_model)
        )

        # Decoder: Stack of DecoderLayers
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        FullAttention(True, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads
                    ),
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads
                    ),
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

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mode):
        """
        Forward pass through the Transformer model.

        Args:
            x_enc (torch.Tensor): Input data for encoding.
            x_mark_enc (torch.Tensor): Time features for encoding.
            x_dec (torch.Tensor): Decoder input data.
            x_mark_dec (torch.Tensor): Time features for decoding.
            mode (str): Mode of operation, 'train' or 'test'.

        Returns:
            torch.Tensor: Model's prediction output.
        """
        # Initialize decoder input with zeros and concatenate with encoded data
        dec_inp = torch.zeros_like(x_dec[:, -self.pred_len:, :]).float()
        dec_inp = torch.cat([x_enc[:, -self.label_len:, :], dec_inp], dim=1).float().to(x_enc.device)

        # Pass input through encoder and decoder
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out = self.encoder(enc_out)

        dec_out = self.dec_embedding(dec_inp, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out)

        return dec_out[:, -self.pred_len:, :]  # Return predicted values for the target sequence
