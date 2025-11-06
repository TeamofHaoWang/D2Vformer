import torch
import torch.nn as nn
from layers.PatchTST_backbone import PatchTST_backbone

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


class T2V_PatchTST(nn.Module):
    """
    T2V (Time-to-Vector) integration with PatchTST model.
    """
    def __init__(self, configs):
        super(T2V_PatchTST, self).__init__()

        # Load parameters from configs
        c_in = configs.d_feature
        context_window = configs.seq_len
        target_window = configs.pred_len
        self.seq_len = context_window
        self.pred_len = target_window

        n_layers = configs.e_layers
        n_heads = configs.n_heads
        d_model = configs.d_model
        d_ff = configs.d_ff
        dropout = configs.dropout
        patch_len = configs.patch_len
        stride = configs.stride

        # Initialize T2V position embedding
        self.T2V_position = SineActivation(d_model, configs.T2V_outmodel, configs.d_feature, d_mark=configs.d_mark)

        # Initialize PatchTST model backbone
        self.model = PatchTST_backbone(
            c_in=c_in, context_window=context_window, target_window=target_window,
            patch_len=patch_len, stride=stride, n_layers=n_layers, d_model=d_model,
            n_heads=n_heads, d_k=None, d_v=None, d_ff=d_ff, norm='BatchNorm',
            attn_dropout=False, dropout=dropout, act='gelu', key_padding_mask='auto',
            padding_var=None, attn_mask=None, res_attention=True, pre_norm=False,
            store_attn=False, pe='zeros', learn_pe=True, fc_dropout=dropout, head_dropout=False,
            padding_patch='end', pretrain_head=False, head_type='flatten', individual=False,
            revin=True, affine=False, subtract_last=False, verbose=False
        )

    def forward(self, batch_x, batch_x_mark, batch_y, batch_y_mark, mode):
        """
        Forward pass through the model.

        Args:
            batch_x (torch.Tensor): Input time-series data [Batch, Input length, Channel]
            batch_x_mark (torch.Tensor): Time features for input [Batch, Input length, Features]
            batch_y (torch.Tensor): Target time-series data (not used in this code)
            batch_y_mark (torch.Tensor): Time features for target (not used in this code)
            mode (str): Mode of operation ('train' or 'test')

        Returns:
            torch.Tensor: Predicted output [Batch, Predicted length, Channel]
        """
        # Generate T2V embeddings using time-related features
        T2V_input = torch.cat((batch_x_mark, batch_y_mark[:, -self.pred_len:, :]), dim=-2)
        T2V_input = T2V_input.transpose(-1, -2)  # [B, D, L]
        T2V_output = self.T2V_position(batch_x.permute(0, 2, 1), T2V_input)

        # Keep k-dimension and apply mean to generate position encoding
        T2V_output = T2V_output.permute(0, 2, 1, 3)
        T2V_x_date = T2V_output[:, :self.seq_len, :, :]
        T2V_x_position = torch.mean(T2V_x_date, dim=-1)

        # Add T2V embedding and pass through PatchTST model
        Transformer_input = T2V_x_position + batch_x
        x = Transformer_input.permute(0, 2, 1)  # [Batch, Channel, Input length]
        x = self.model(x)
        x = x.permute(0, 2, 1)  # [Batch, Input length, Channel]

        return x  # Return the model's output
