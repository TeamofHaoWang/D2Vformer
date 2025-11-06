import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.iTransformer_EncDec import Encoder, EncoderLayer
from layers.iTransformer_SelfAtt_Famliy import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted

def t2v(tau, f, out_features, w, b, w0, b0, arg=None):
    """
    Generate time-series position embeddings (trend + periodic components).

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
    Sine activation function for time-series data.
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


class T2V_iTransformer(nn.Module):
    """
    iTransformer model with T2V (Time-to-Vector) integration for position encoding.
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, configs):
        super(T2V_iTransformer, self).__init__()

        # Initialize model parameters
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = False
        self.use_norm = True

        # T2V position embedding
        self.T2V_position = SineActivation(configs.d_model, configs.T2V_outmodel, configs.d_feature, d_mark=configs.d_mark)

        # Input embedding
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq, configs.dropout)

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
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        # Output projection
        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        Forecasting function for time-series data with T2V encoding.

        Args:
            x_enc (torch.Tensor): Encoded input sequence.
            x_mark_enc (torch.Tensor): Time features for input.
            x_dec (torch.Tensor): Decoding input (unused here).
            x_mark_dec (torch.Tensor): Time features for decoding (unused here).

        Returns:
            torch.Tensor: The predicted output.
        """
        if self.use_norm:
            # Normalize the input data
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        _, _, N = x_enc.shape  # B L N

        # Generate T2V embeddings
        T2V_input = torch.cat((x_mark_enc, x_mark_dec[:, -self.pred_len:, :]), dim=-2)
        T2V_input = T2V_input.transpose(-1, -2)  # B D L
        T2V_output = self.T2V_position(x_enc.permute(0, 2, 1), T2V_input)

        # Keep k-dimension and apply mean to generate position encoding
        T2V_output = T2V_output.permute(0, 2, 1, 3)
        T2V_x_date = T2V_output[:, :self.seq_len, :, :]
        T2V_x_position = torch.mean(T2V_x_date, dim=-1)

        # Combine with original input for transformer
        Transformer_input = T2V_x_position + x_enc

        # Embedding and pass through encoder
        enc_out = self.enc_embedding(Transformer_input, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Project the encoder output to prediction space
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N]  # Filter covariates

        if self.use_norm:
            # De-normalize output
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mode):
        """
        Forward pass through the model.

        Args:
            x_enc (torch.Tensor): Encoded input sequence [B, L, D]
            x_mark_enc (torch.Tensor): Time features for input
            x_dec (torch.Tensor): Decoding input (unused in this model)
            x_mark_dec (torch.Tensor): Time features for decoding
            mode (str): Mode of operation ('train' or 'test')

        Returns:
            torch.Tensor: Predicted output [B, L, D]
        """
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        pred = dec_out[:, -self.pred_len:, :]
        return pred  # [B, L, D]
