from torch import nn
from layers.PatchTST_backbone import PatchTST_backbone
from layers.PatchTST_layers import series_decomp
import torch
import matplotlib.pyplot as plt
import numpy as np
import os


class Date2Vec_Fourier(nn.Module):
    def __init__(self, in_features, out_features, d_features, d_mark, save_path):
        # For D2V, in_features refers to seq_len and out_features refers to the number of sinusoids used to characterize each D
        super(Date2Vec_Fourier, self).__init__()
        self.out_features = out_features

        self.f = torch.cos
        self.dominance_freq = int(in_features // 24 + 1) * 2 + 10
        self.freq_upsampler_real = nn.Linear(self.dominance_freq, out_features)  # Complex layer for frequency upsampling
        self.freq_upsampler_imag = nn.Linear(self.dominance_freq, out_features)  # Complex layer for frequency upsampling
        self.top_k = 8
        print("Using topK")

        self.w_fourier = torch.fft.fftfreq(out_features, 1) * 2 * torch.pi
        self.d2v_vision_flag = True
        self.save_path = save_path
        print('Using Date2Vec_Fourier')

    def forward(self, data, tau, mode):
        # Compute the Fourier Transform of the input data
        low_specx = torch.fft.rfft(data, dim=-1)
        if low_specx.shape[1] < self.dominance_freq:
            low_specx = torch.fft.rfft(data, n=2 * (self.dominance_freq - 1), dim=-1)  # n specifies the length of the Fourier transform

        low_specx = torch.view_as_real(low_specx[:, :, 0:self.dominance_freq])  # Convert complex values to real values as 2D vectors
        low_specx_real = low_specx[:, :, :, 0]
        low_specx_imag = low_specx[:, :, :, 1]

        # Upsample the real and imaginary parts separately
        real = self.freq_upsampler_real(low_specx_real)
        imag = self.freq_upsampler_imag(low_specx_imag)
        low_specxy_real = real - imag
        low_specxy_imag = real + imag

        # Zero-padding to the specified output size
        low_specxy_R = torch.zeros([low_specxy_real.size(0), low_specxy_real.size(1), self.out_features], dtype=low_specxy_real.dtype).to(low_specxy_real.device)
        low_specxy_R[:, :, 0:low_specxy_real.size(2)] = low_specxy_real

        low_specxy_I = torch.zeros([low_specxy_imag.size(0), low_specxy_real.size(1), self.out_features], dtype=low_specxy_imag.dtype).to(low_specxy_imag.device)
        low_specxy_I[:, :, 0:low_specxy_real.size(2)] = low_specxy_imag

        # Combine the real and imaginary parts to form complex numbers
        low_specxy = torch.complex(low_specxy_R, low_specxy_I)
        scale_attitude = torch.full((low_specxy_real.size(2),), 2 / low_specxy_real.size(2)).to(low_specxy_imag.device)
        scale_attitude[0] = 1 / low_specxy_real.size(2)
        scale_attitude.unsqueeze(0).unsqueeze(0).repeat(low_specxy_imag.size(0), low_specxy_imag.size(1), 1)

        # Calculate the amplitude and phase
        attitude = torch.sqrt(torch.pow(low_specxy_I, 2) + torch.pow(low_specxy_R, 2)) * scale_attitude
        theta = torch.angle(low_specxy)

        output = self.D2V(attitude, theta, tau, self.f, mode)
        return output

    def D2V(self, attitude, theta, tau, f, mode):
        _, D, _ = attitude.shape
        tau = tau[:, 0, :].unsqueeze(-1).repeat(1, 1, D)  # (B, L+O, d_feature)
        tau = tau.transpose(1, -1)  # (B, d_feature, L+O)
        theta = theta.unsqueeze(2)
        attitude_topk, _ = self.keep_topk(attitude, self.top_k)
        attitude_topk = attitude_topk.unsqueeze(2)
        attitude = attitude.unsqueeze(2)
        tau = tau.unsqueeze(2).transpose(2, -1)

        # Fourier transformation applied to tau
        w_fourier = self.w_fourier.unsqueeze(-1).to(tau.device)
        w_tau = torch.einsum('bdln,fn->bdlf', tau, w_fourier)
        v1 = attitude * f(w_tau + theta)

        # Visualization for testing mode
        if mode == "test" and self.d2v_vision_flag == False:
            self.plot_cos_waves(tau[0, 0, :, 0], tau.shape[2], w_fourier[:, 0], attitude_topk[0, 0, 0, :])
            self.d2v_vision_flag = False

        return v1

    def keep_topk(self, tensor, k):
        # Get the indices of the top k values in tensor
        _, indices = torch.topk(tensor, k, dim=-1)

        # Create a mask tensor with zeros
        mask = torch.zeros_like(tensor)

        # Set the positions of the top k values to the original tensor values, others set to 0
        mask.scatter_(-1, indices, 1)

        # Multiply the input tensor by the mask to retain top k values and set others to 0
        output = tensor * mask.float()

        return output, indices

    def plot_cos_waves(self, time_range, num_points, frequencies, amplitudes):
        # Plot multiple cosine waves with different frequencies and amplitudes
        time_range = time_range.cpu().detach().numpy()
        frequencies = frequencies.cpu().detach().numpy()
        amplitudes = amplitudes.cpu().detach().numpy()

        t = time_range
        t_index = np.linspace(1, time_range.shape[0], num_points)

        fig, axes = plt.subplots(self.top_k, 1, figsize=(10, 8), sharex=True)

        i = 0
        for f, a in zip(frequencies, amplitudes):
            if a != 0:
                y_wave = a * np.cos(f * t)
                axes[i].plot(t_index, y_wave)
                axes[i].set_title(f'Frequency: {f} Hz, Amplitude: {a}')
                axes[i].set_ylabel('Amplitude')
                i += 1

        axes[-1].set_xlabel('Time (s)')

        # Adjust the subplots for better spacing
        plt.subplots_adjust(hspace=0.5)
        #plt.show()
        plt.savefig(self.save_path + '_multi_cos_vision.png')
        print(f"Date2Vec image saved")

class D2V_Fourier_PatchTST(nn.Module):
    def __init__(self, configs):
        super().__init__()

        # Load parameters from configuration
        c_in = configs.d_feature
        context_window = configs.seq_len
        target_window = configs.pred_len

        n_layers = configs.e_layers
        n_heads = configs.n_heads
        d_model = configs.d_model
        d_ff = configs.d_ff
        dropout = configs.dropout
        patch_len = configs.patch_len
        stride = configs.stride
        self.mark_index = configs.mark_index

        # Initialize the PatchTST backbone
        self.model = PatchTST_backbone(c_in=c_in, context_window=context_window, target_window=target_window,
                                       patch_len=patch_len, stride=stride, n_layers=n_layers, d_model=d_model,
                                       n_heads=n_heads, d_k=None, d_v=None, d_ff=d_ff, norm='BatchNorm',
                                       attn_dropout=False, dropout=dropout, act='gelu', key_padding_mask='auto',
                                       padding_var=None, attn_mask=None, res_attention=True, pre_norm=False,
                                       store_attn=False, pe='zeros', learn_pe=True, fc_dropout=dropout,
                                       head_dropout=False, padding_patch='end', pretrain_head=False,
                                       head_type='flatten', individual=False, revin=True, affine=False,
                                       subtract_last=False, verbose=False)

        # Initialize Date2Vec for position encoding
        self.D2V_position = Date2Vec_Fourier(context_window, d_model, c_in, d_mark=len(self.mark_index), save_path=configs.output_path)

    def forward(self, batch_x, batch_x_mark, batch_y, batch_y_mark, mode):
        # Rearrange the input batch dimensions
        x = batch_x.permute(0, 2, 1)  # x: [Batch, Channel, Input length]
        batch_x_mark = batch_x_mark[:, :, self.mark_index]  # Select the relevant marks

        # Apply Date2Vec position encoding
        D2V_output = self.D2V_position(x, batch_x_mark.permute(0, 2, 1), mode)
        x = x + torch.sum(D2V_output, dim=-1)

        # Pass through the PatchTST model
        x = self.model(x)
        x = x.permute(0, 2, 1)  # x: [Batch, Input length, Channel]

        return x
