import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
import math

def t2v(tau, f, out_features, w, b, w0, b0, arg=None):
    """
    Generate time-series position embeddings by combining trend and periodic components.

    Args:
        tau (torch.Tensor): Input tensor.
        f (function): Activation function (e.g., sine).
        out_features (int): Output feature size.
        w (torch.Tensor): Weight matrix for periodic term.
        b (torch.Tensor): Bias for periodic term.
        w0 (torch.Tensor): Weight matrix for trend term.
        b0 (torch.Tensor): Bias for trend term.
        arg: Optional argument for activation function.

    Returns:
        torch.Tensor: Concatenated trend and periodic components.
    """
    tau = tau.transpose(-1, -2)
    if arg:
        v1 = f(torch.matmul(tau, w) + b, arg)
    else:
        v1 = f(torch.matmul(tau, w) + b)  # Periodic term
    v2 = torch.matmul(tau, w0) + b0  # Trend term
    return torch.cat([v2, v1], -1)  # Concatenate trend and periodic terms


class SineActivation(nn.Module):
    """
    Sine activation function for time-series data to generate embeddings.
    """
    def __init__(self, in_features, out_features, d_features, d_mark):
        super(SineActivation, self).__init__()

        # Initialize parameters for sine-based transformation
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(d_mark, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(1))
        self.w = nn.parameter.Parameter(torch.randn(d_mark, out_features - 1))
        self.b = nn.parameter.Parameter(torch.randn(out_features - 1))
        self.f = torch.sin
        self.d_features = d_features

        # Initialize parameters
        nn.init.uniform_(self.w0, -1, 1)
        nn.init.uniform_(self.b0, -1, 1)
        nn.init.uniform_(self.w, -1, 1)
        nn.init.uniform_(self.b, -1, 1)

    def forward(self, data, tau, flag=None):
        """
        Apply the sine activation function to the input data.

        Args:
            data (torch.Tensor): Input data.
            tau (torch.Tensor): Time-related information.
            flag: Optional flag for specific operations.

        Returns:
            torch.Tensor: Transformed output with time-series embeddings.
        """
        output = t2v(tau, self.f, self.out_features, self.w, self.b, self.w0, self.b0)
        output = output.unsqueeze(1).repeat(1, self.d_features, 1, 1)
        return output


class T2V_Transformer(nn.Module):
    """
    T2V (Time-to-Vector) used as position encoding in Transformer.
    """
    def __init__(self, configs):
        super(T2V_Transformer, self).__init__()

        # Model configuration
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.label_len = configs.label_len
        self.d_mark = configs.d_mark
        self.output_attention = False
        self.d_feature = configs.d_feature
        self.d_model = configs.d_model

        # Input projection
        self.transformer_proj = nn.Linear(self.d_feature, self.d_model)

        # Output projection
        self.output_proj = nn.Linear(self.seq_len, self.pred_len)

        # T2V position embedding
        self.T2V_position = SineActivation(self.d_model, configs.T2V_outmodel, self.d_feature, d_mark=configs.d_mark)

        # Encoder
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

        # Decoder
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

    def forward(self, x_enc, x_mark_enc, y_batch, x_mark_dec, mode):
        """
        Forward pass through the model.

        Args:
            x_enc (torch.Tensor): Encoded input sequence.
            x_mark_enc (torch.Tensor): Time features for input.
            y_batch (torch.Tensor): Target data (unused in this code).
            x_mark_dec (torch.Tensor): Time features for target (unused in this code).
            mode (str): Mode of operation ('train' or 'test').

        Returns:
            torch.Tensor: Predicted output.
        """
        B, L, D = x_enc.shape

        # Generate T2V embeddings using time-related features
        T2V_input = torch.cat((x_mark_enc, x_mark_dec[:, -self.pred_len:, :]), dim=-2)
        T2V_input = T2V_input.transpose(-1, -2)  # [B, D, L]
        T2V_output = self.T2V_position(x_enc.permute(0, 2, 1), T2V_input)

        # Keep k-dimension and apply mean to generate position encoding
        T2V_output = T2V_output.permute(0, 2, 1, 3)
        T2V_x_date = T2V_output[:, :self.seq_len, :, :]
        T2V_x_position = torch.mean(T2V_x_date, dim=-1)

        # Add T2V embedding and pass through Transformer
        Transformer_input = T2V_x_position + x_enc
        Transformer_input = self.transformer_proj(Transformer_input)

        encoder_output = self.encoder(Transformer_input)
        decoder_output = self.decoder(Transformer_input, encoder_output)

        output = self.output_proj(decoder_output.permute(0, 2, 1)).permute(0, 2, 1)

        return output
