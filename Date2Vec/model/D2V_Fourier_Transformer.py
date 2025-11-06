import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from layers.Date2Vec import Date2Vec_Fourier
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
import os

plt.rcParams['font.sans-serif'] = ['SimHei']  # Used for displaying Chinese labels correctly
plt.rcParams['axes.unicode_minus'] = False  # Used for displaying negative signs correctly

class D2V_Fourier_Transformer(nn.Module):
    """
    D2V as the position in the Transformer.
    """
    def __init__(self, configs):
        super(D2V_Fourier_Transformer, self).__init__()

        # Initialize model parameters from config
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.label_len = configs.label_len
        self.d_mark = configs.d_mark
        self.d_feature = configs.d_feature
        self.d_model = configs.d_model
        self.input_mark = 2
        self.mark_index = configs.mark_index

        # Date2Vec for encoding the historical time information
        self.linear_t = nn.Linear(self.seq_len, self.d_model)
        self.linear_t_out = nn.Linear(self.d_model, self.seq_len)

        # Project input features to model dimension
        self.transformer_proj = nn.Linear(self.d_feature, self.d_model)

        # Output projection layer for prediction
        self.output_proj = nn.Linear(self.seq_len, self.pred_len)

        # Date2Vec Fourier for position encoding
        self.D2V_position = Date2Vec_Fourier(self.seq_len, configs.T2V_outmodel, self.d_feature,
                                             d_mark=len(self.mark_index), save_path=configs.output_path)

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

    def forward(self, x_enc, x_mark_enc, y_batch, x_mark_dec, mode):
        """
        Forward pass of the model.

        Args:
            x_enc: Encoded input tensor (B, L, D)
            x_mark_enc: Encoded time features for input
            y_batch: Actual batch data for prediction (not used in this code)
            x_mark_dec: Decoder time features
            mode: Mode of operation ('norm' for normalization, 'denorm' for denormalization)

        Returns:
            output: Final output after decoding
        """
        B, L, D = x_enc.shape

        # Prepare Date2Vec input by selecting the relevant time features
        D2V_input = x_mark_enc[:, :, self.mark_index].transpose(-1, -2)

        # Get Date2Vec output for positional encoding
        D2V_output = self.D2V_position(x_enc.permute(0, 2, 1), D2V_input, mode)

        # Sum along the k-dimension for positional representation
        D2V_x_position = torch.sum(D2V_output, dim=-1)

        # Combine Date2Vec positional encoding with the original input for Transformer input
        Transformer_input = D2V_x_position + x_enc

        # Map input to higher dimensions for Transformer
        Transformer_input = self.transformer_proj(Transformer_input)

        # Pass the input through the encoder and decoder
        encoder_output = self.encoder(Transformer_input)
        decoder_output = self.decoder(Transformer_input, encoder_output)

        # Project the decoder output to the desired prediction length
        output = self.output_proj(decoder_output.permute(0, 2, 1)).permute(0, 2, 1)

        return output
