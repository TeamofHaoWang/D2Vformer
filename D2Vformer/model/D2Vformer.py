import torch
import torch.nn as nn
import math
from layers.koopa_related import *
from layers.koopa_AR_related import *
from layers.Date2Vec import *
from layers.Fusion_Block import *
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class D2Vformer(nn.Module):
    def __init__(self, configs):
        super(D2Vformer, self).__init__()

        # Initialize model parameters from configs
        self.configs = configs
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.d_feature = configs.d_feature
        self.d_model = configs.d_model

        # Encoder layers for time-invariant components
        self.all_linear = nn.Linear(self.seq_len, self.d_model)
        self.season_linear = nn.Linear(self.seq_len, self.d_model)
        self.trend_linear_decoder = nn.Linear(self.d_model, self.pred_len)

        # Mapping time-invariant data using stable initialization
        K_init = torch.randn(self.d_model, self.d_model)
        U, _, V = torch.svd(K_init)
        self.K = nn.Linear(self.d_model, self.d_model, bias=False)
        self.K.weight.data = torch.mm(U, V.t())

        # Decoder layers
        self.project = nn.Linear(self.d_model, self.pred_len)

        # Encoding of time information
        self.linear_t = nn.Linear(self.seq_len, self.d_model)
        self.linear_t_out = nn.Linear(self.d_model, self.seq_len)

        # Revin for normalization
        self.revin = RevIN(self.d_feature, affine=False, subtract_last=False)

        # Date2Vec Fourier for time-series encoding
        self.Date2Vec = Date2Vec_Fourier(self.d_model, configs.T2V_outmodel, self.d_feature,
                                         d_mark=configs.d_mark, save_path=configs.save_path)

        # Fusion block
        self.fusion = Fusion_Block(args=configs, save_path=configs.save_path)

        # Patch-related parameters
        self.patch_len = configs.patch_len
        self.stride = configs.stride

        # Fourier decomposition for time-series data
        self.mask_spectrum = configs.mask_spectrum
        self.disentanglement = FourierFilter(self.mask_spectrum)

        # BatchNorm layers
        self.bn_inv = nn.BatchNorm1d(self.d_model)
        self.bn_var = nn.BatchNorm1d(self.d_feature)
        self.bn_pred = nn.BatchNorm1d(self.d_feature)

    def do_patching(self, z, stride):
        """Patches the input tensor to ensure proper segmenting"""
        z = nn.functional.pad(z, (0, stride))  # Padding
        z = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)  # Perform patching
        return z

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mode):
        """
        Forward pass of the model

        Args:
            x_enc (torch.Tensor): Encoded input tensor
            x_mark_enc (torch.Tensor): Encoded date/time features for input
            x_dec (torch.Tensor): Decoder input tensor
            x_mark_dec (torch.Tensor): Decoder date/time features
            mode (str): Mode of operation (e.g., 'train', 'test')

        Returns:
            prediction (torch.Tensor): The predicted output
        """
        # Apply Revin normalization to input
        x_enc = self.revin(x_enc, 'norm')

        # Decompose input into time-varying and time-invariant components
        x_var, x_inv = self.disentanglement(x_enc)

        # Process time-invariant components
        trend_output = self.all_linear(x_inv.permute(0, 2, 1)).permute(0, 2, 1)
        season_output = self.season_linear(x_inv.permute(0, 2, 1)).permute(0, 2, 1)
        encoder_output = trend_output + season_output  # Combine trend and season outputs
        encoder_output = self.bn_inv(encoder_output)

        # Apply global K operator on the time-invariant data
        y_inv = self.K(encoder_output.transpose(-1, -2)).transpose(-1, -2)
        y_inv = self.project(y_inv.permute(0, 2, 1)).permute(0, 2, 1)

        # Process time-varying data
        D2V_input = torch.cat((x_mark_enc, x_mark_dec[:, -self.pred_len:, :]), dim=-2)
        D2V_input = D2V_input.transpose(-1, -2)

        # Apply linear transformations to time-varying data
        all_output_var = self.all_linear(x_var.permute(0, 2, 1))
        season_output_var = self.season_linear(x_var.permute(0, 2, 1))
        trend_output_var = self.trend_linear_decoder(all_output_var - season_output_var).transpose(-1, -2)

        # Normalize time-varying data
        encoder_output_var = season_output_var
        encoder_output_var = self.bn_var(encoder_output_var)

        # Date2Vec encoding of historical and predicted data
        D2V_output = self.Date2Vec(encoder_output_var, D2V_input, mode)
        D2V_output = D2V_output.permute(0, 2, 1, 3)

        # Extract encoded date features for the model
        D2V_x_date = D2V_output[:, :self.seq_len, :, :]
        D2V_y_date = D2V_output[:, -self.pred_len:, :, :]

        # Perform patching on both the input data and date features
        x_var = self.do_patching(x_var.transpose(-1, 1), self.configs.stride)
        D2V_x_date = self.do_patching(D2V_x_date.transpose(1, -1), self.configs.stride)
        D2V_y_date = self.do_patching(D2V_y_date.transpose(1, -1), self.configs.stride)

        # Reshape the patched data for fusion
        B, k, D, P_N, P_L = D2V_x_date.shape
        D2V_x_date = D2V_x_date.permute(0, 2, 3, 4, 1).reshape(B, P_N, D, P_L * k)
        B, k, D, P_N, P_L = D2V_y_date.shape
        D2V_y_date = D2V_y_date.permute(0, 2, 3, 4, 1).reshape(B, P_N, D, P_L * k)

        # Fusion of time-varying and time-invariant data
        prediction_var = self.fusion(x_var, D2V_x_date, D2V_y_date, mode)

        # Combine all outputs for final prediction
        prediction = prediction_var + y_inv + trend_output_var
        prediction = self.bn_pred(prediction.transpose(-1, -2)).transpose(-1, -2)

        # Apply denormalization
        prediction = self.revin(prediction, 'denorm')

        return prediction  # Final predicted output
