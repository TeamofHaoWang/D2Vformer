import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams['font.sans-serif'] = ['SimHei']  # To display Chinese labels correctly
plt.rcParams['axes.unicode_minus'] = False  # To display negative signs correctly

class PositionalEmbedding(nn.Module):
    def __init__(self, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space
        pe = torch.zeros(max_len).float()
        pe.requires_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, 1, 2).float() * -(math.log(10000.0) / 1)).exp()

        pe = torch.sin(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:x.size(-1), :]


class DataEmbedding_onlypos(nn.Module):
    def __init__(self, dropout=0.1):
        super(DataEmbedding_onlypos, self).__init__()

        self.position_embedding = PositionalEmbedding()

    def forward(self, x):
        x = self.position_embedding(x)
        return x.permute(1, 0)


class Transpose(nn.Module):
    """
    Custom layer to swap tensor dimensions.
    """

    def __init__(self, dims):
        """
        Initialize function.

        Parameters:
        dims (list or tuple): The list or tuple of dimension indices to swap.
        """
        super(Transpose, self).__init__()
        self.dims = dims

    def forward(self, x):
        """
        Forward pass function to swap the dimensions of the input tensor.

        Parameters:
        x (torch.Tensor): Input tensor.

        Returns:
        torch.Tensor: Tensor with swapped dimensions.
        """
        return x.transpose(*self.dims)


class Date2Vec(nn.Module):
    def __init__(self, in_features, out_features, d_features, d_mark):
        # For D2V, the input in_features refers to seq_len and
        # out_features refers to how many sinusoids are used to characterize each D
        super(Date2Vec, self).__init__()
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(d_features, 1, 1, 1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        nn.init.uniform_(self.w, 0, 2 * torch.pi)
        self.b = nn.parameter.Parameter(torch.randn(d_features, 1, 1, out_features - 1))
        self.f = torch.sin
        self.w_transform_t = nn.Parameter(torch.randn((d_mark, d_features)), requires_grad=True)
        self.w_transform_s = nn.Parameter(torch.randn((d_mark, d_features)), requires_grad=True)

    def forward(self, data, tau):
        output = self.D2V(data, tau, self.f, self.w, self.b, self.w0, self.b0)
        return output

    def D2V(self, data, tau, f, w, b, w0, b0):
        _, D, _ = data.shape
        tau = tau.unsqueeze(-1).repeat(1, 1, 1, D)  # (B, d_mark, L+O, d_feature)
        tau = tau.transpose(1, -1)  # (B, d_feature, L+O, d_mark)
        mark = tau.shape[-1]
        w_trend = torch.matmul(data, w0).unsqueeze(-2)
        w_season = torch.matmul(data, w).unsqueeze(-2)

        w_trend = w_trend.unsqueeze(2).repeat(1, 1, mark, 1, 1)
        w_season = w_season.unsqueeze(2).repeat(1, 1, mark, 1, 1)
        tau = tau.unsqueeze(2).transpose(2, -1)
        v2 = torch.matmul(tau, w_trend) + b0
        v1 = f(torch.matmul(tau, w_season) + b)

        self.w_trend = w_trend
        self.w_season = w_season
        # Melting away the mark dimension (B, D, L, k)
        v1 = torch.mean(v1, dim=2)
        v2 = torch.mean(v2, dim=2)

        return torch.cat([v2, v1], -1)


class Date2Vec_Fourier(nn.Module):
    def __init__(self, in_features, out_features, d_features, d_mark, save_path):
        # For D2V, the input in_features refers to seq_len and
        # out_features refers to how many sinusoids are used to characterize each D
        super(Date2Vec_Fourier, self).__init__()
        self.out_features = out_features

        self.f = torch.cos
        self.dominance_freq = int(in_features // 24 + 1) * 2 + 10
        self.freq_upsampler_real = nn.Linear(self.dominance_freq,
                                             out_features)  # Complex layer for frequency upsampling
        self.freq_upsampler_imag = nn.Linear(self.dominance_freq,
                                             out_features)  # Complex layer for frequency upsampling
        self.top_k = 8
        print("Using topK")

        # Mark fusion
        self.mark_fusion = nn.Linear(d_mark, 1)

        self.w_fourier = torch.normal(mean=1.0, std=1.0, size=(out_features,))
        self.d2v_vision_flag = True
        self.save_path = save_path
        print('Using Date2Vec_Fourier')

    def forward(self, data, tau, mode):
        low_specx = torch.fft.rfft(data, dim=-1)
        if low_specx.shape[1] < self.dominance_freq:
            low_specx = torch.fft.rfft(data, n=2 * (self.dominance_freq - 1), dim=-1)  # n represents the FFT length

        low_specx = torch.view_as_real(low_specx[:, :, 0:self.dominance_freq])  # Convert complex values into 2D vector
        low_specx_real = low_specx[:, :, :, 0]
        low_specx_imag = low_specx[:, :, :, 1]

        real = self.freq_upsampler_real(low_specx_real)
        imag = self.freq_upsampler_imag(low_specx_imag)
        low_specxy_real = real - imag
        low_specxy_imag = real + imag

        low_specxy_R = torch.zeros(
            [low_specxy_real.size(0), low_specxy_real.size(1), self.out_features],
            dtype=low_specxy_real.dtype).to(low_specxy_real.device)  # Padding to specific length
        low_specxy_R[:, :, 0:low_specxy_real.size(2)] = low_specxy_real

        low_specxy_I = torch.zeros(
            [low_specxy_imag.size(0), low_specxy_real.size(1), self.out_features],
            dtype=low_specxy_imag.dtype).to(low_specxy_imag.device)  # Padding to specific length
        low_specxy_I[:, :, 0:low_specxy_real.size(2)] = low_specxy_imag

        low_specxy = torch.complex(low_specxy_R, low_specxy_I)
        # Scale for attitude
        scale_attitude = torch.full((low_specxy_real.size(2),), 2 / low_specxy_real.size(2)).to(low_specxy_imag.device)
        scale_attitude[0] = 1 / low_specxy_real.size(2)
        scale_attitude = scale_attitude.unsqueeze(0).unsqueeze(0).repeat(low_specxy_imag.size(0), low_specxy_imag.size(1), 1)

        # Phase angle = arctan(imag/real)  Magnitude = sqrt(real^2 + imag^2)
        attitude = torch.sqrt(torch.pow(low_specxy_I, 2) + torch.pow(low_specxy_R, 2)) * scale_attitude
        theta = torch.angle(low_specxy)

        output = self.D2V(attitude, theta, tau, self.f, mode)
        return output

    def D2V(self, attitude, theta, tau, f, mode):
        _, D, _ = attitude.shape
        tau = self.mark_fusion(tau.transpose(1, -1)).repeat(1, 1, D)  # (B, L+O, d_feature)
        tau = tau.transpose(1, -1)  # (B, d_feature, L+O)
        theta = theta.unsqueeze(2)

        attitude_topk, _ = self.keep_topk(attitude, self.top_k)
        attitude_topk = attitude_topk.unsqueeze(2)
        attitude_topk.detach().requires_grad = False

        attitude = attitude.unsqueeze(2)
        tau = tau.unsqueeze(2).transpose(2, -1)

        w_fourier = self.w_fourier.unsqueeze(-1).to(tau.device)
        w_tau = torch.einsum('bdln,fn->bdlf', tau, w_fourier)
        v1 = attitude * f(w_tau)

        vision_input = v1.clone()
        # Vision plotting
        if mode == "test" and self.d2v_vision_flag == True:
            self.plot_cos_waves(vision_input[0, 0, :, :])
            self.d2v_vision_flag = False

        return v1

    def keep_topk(self, tensor, k):
        """
        Retain the top k largest values in the tensor and set others to 0

        Parameters:
        tensor (torch.Tensor): Input tensor
        k (int): Number of top largest values to retain

        Returns:
        torch.Tensor: Tensor with only the top k largest values retained
        """
        # Get the top k largest values indices
        _, indices = torch.topk(tensor, k, dim=-1)

        # Create a zero tensor of the same shape as the input tensor
        mask = torch.zeros_like(tensor)

        # Set the top k largest values to the original tensor's corresponding values, others to 0
        mask.scatter_(-1, indices, 1)

        # Multiply the input tensor by the mask to retain only the top k largest values
        output = tensor * mask.float()

        return output, indices

    def plot_cos_waves(self, v1):
        """
        Plot multiple sine waves with different frequencies and amplitudes on the same graph.

        Parameters:
        v1 (torch.Tensor): Input tensor containing the sine wave data
        """
        # 1. Compute the mean for each feature (column)
        feature_mean_vals = torch.mean(v1, dim=0)  # Get the mean for each column

        # 2. Get the top 8 features with the highest mean values
        _, top_k_indices = torch.topk(feature_mean_vals, self.top_k)  # Get indices of the top 8 mean values

        # 3. Extract the time series for these top 8 features
        top_k_features = v1[:, top_k_indices]  # Extract the top 8 features by index

        # Create subplots
        fig, axes = plt.subplots(self.top_k, 1, figsize=(10, 8), sharex=True)

        # Plot each sine wave
        for j in range(self.top_k):
            y_wave = top_k_features[:, j].detach().cpu()
            axes[j].plot(y_wave)
            axes[j].set_title(f'Feature {top_k_indices[j].item()}')
            axes[j].set_ylabel('Amplitude')
        # Add x-axis label
        axes[-1].set_xlabel('Time (seconds)')
        # Adjust subplots and display
        plt.subplots_adjust(hspace=0.5)
        # plt.show()
        plt.savefig(self.save_path + '_multi_cos_vision.png')
        print(f"Date2Vec image has been saved")
