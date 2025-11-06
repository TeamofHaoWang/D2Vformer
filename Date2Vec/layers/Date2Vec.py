import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
class PositionalEmbedding(nn.Module):
    def __init__(self, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, 1, 2).float() * -(math.log(10000.0) / 1)).exp()

        pe = torch.sin(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[ :x.size(-1),:]


class DataEmbedding_onlypos(nn.Module):
    def __init__(self,  dropout=0.1):
        super(DataEmbedding_onlypos, self).__init__()

        self.position_embedding = PositionalEmbedding()


    def forward(self, x):
        x = self.position_embedding(x)
        return x.permute(1,0)

class Transpose(nn.Module):
    """
    交换 tensor 维度的自定义层。
    """

    def __init__(self, dims):
        """
        初始化函数。

        参数:
        dims (list or tuple): 需要交换的维度索引列表或元组。
        """
        super(Transpose, self).__init__()
        self.dims = dims

    def forward(self, x):
        """
        前向传播函数,交换输入 tensor 的维度。

        参数:
        x (torch.Tensor): 输入 tensor。

        返回:
        torch.Tensor: 维度交换后的 tensor。
        """
        return x.transpose(*self.dims)

class Date2Vec(nn.Module):
    def __init__(self, in_features, out_features, d_features,d_mark):
        # For D2V, here the input in_features refers to seq_len and
        # out_features refers to how many sinusoids are used to characterise the corresponding each D
        super(Date2Vec, self).__init__()
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(d_features, 1, 1, 1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        nn.init.uniform_(self.w, 0, 2 * torch.pi)
        self.b = nn.parameter.Parameter(torch.randn(d_features, 1, 1, out_features - 1))
        self.f = torch.sin
        self.w_transform_t=nn.Parameter(torch.randn((d_mark,d_features)),requires_grad=True)
        self.w_transform_s = nn.Parameter(torch.randn((d_mark,d_features)), requires_grad=True)

    def forward(self, data, tau):
        output=self.D2V(data, tau, self.f, self.w, self.b, self.w0, self.b0)
        return output

    def D2V(self,data, tau, f, w, b, w0, b0):
        _,D,_=data.shape
        tau=tau.unsqueeze(-1).repeat(1,1,1,D)#(B,d_mark,L+O,d_feature)
        tau=tau.transpose(1,-1)#(B,d_feature,L+O,d_mark)
        mark=tau.shape[-1]
        w_trend = torch.matmul(data, w0).unsqueeze(-2)
        w_season = torch.matmul(data, w).unsqueeze(-2)

        w_trend = w_trend.unsqueeze(2).repeat(1,1,mark,1,1)
        w_season = w_season.unsqueeze(2).repeat(1, 1, mark, 1, 1)
        tau = tau.unsqueeze(2).transpose(2, -1)
        v2 = torch.matmul(tau, w_trend) + b0
        v1 = f(torch.matmul(tau, w_season) + b)

        self.w_trend=w_trend
        self.w_season=w_season
        # Melting away the mark dimension (B,D,L,k)
        v1 = torch.mean(v1,dim=2)
        v2 = torch.mean(v2,dim=2)

        return torch.cat([v2, v1], -1)




class Date2Vec_Fourier(nn.Module):
    def __init__(self, in_features, out_features, d_features, d_mark, save_path):
        # For D2V, here the input in_features refers to seq_len and
        # out_features refers to how many sinusoids are used to characterise the corresponding each D
        super(Date2Vec_Fourier, self).__init__()
        self.out_features = out_features

        self.f = torch.cos
        self.dominance_freq = int(in_features // 24 + 1) * 2 + 10
        self.freq_upsampler_real = nn.Linear(self.dominance_freq,
                                             out_features)  # complex layer for frequency upcampling]
        self.freq_upsampler_imag = nn.Linear(self.dominance_freq,
                                             out_features)  # complex layer for frequency upcampling]
        # nn.init.uniform_(self.freq_upsampler_real.weight, 0, 1)
        # nn.init.uniform_(self.freq_upsampler_imag.weight, 0, 1)
        self.top_k = 8
        print("use topK")

        # mark fusion
        self.mark_fusion = nn.Linear(d_mark,1)

        self.w_fourier = torch.normal(mean=1.0, std=1.0, size=(out_features,))
        self.d2v_vision_flag = True
        self.save_path = save_path
        print('Using Date2Vec_Fourier')

    def forward(self, data, tau, mode):
        low_specx = torch.fft.rfft(data, dim=-1)
        if low_specx.shape[1] < self.dominance_freq:
            low_specx = torch.fft.rfft(data, n=2 * (self.dominance_freq - 1), dim=-1)  # n 表示傅里叶域的变换长度

        low_specx = torch.view_as_real(low_specx[:, :, 0:self.dominance_freq])  # 复数值 按 real值存储变成2维向量
        low_specx_real = low_specx[:, :, :, 0]
        low_specx_imag = low_specx[:, :, :, 1]

        real = self.freq_upsampler_real(low_specx_real)
        imag = self.freq_upsampler_imag(low_specx_imag)
        low_specxy_real = real - imag
        low_specxy_imag = real + imag

        low_specxy_R = torch.zeros(
            [low_specxy_real.size(0), low_specxy_real.size(1), self.out_features],
            dtype=low_specxy_real.dtype).to(low_specxy_real.device)  # 填充至特定长度
        low_specxy_R[:, :, 0:low_specxy_real.size(2)] = low_specxy_real

        low_specxy_I = torch.zeros(
            [low_specxy_imag.size(0), low_specxy_real.size(1), self.out_features],
            dtype=low_specxy_imag.dtype).to(low_specxy_imag.device)  # 填充至特定长度
        low_specxy_I[:, :, 0:low_specxy_real.size(2)] = low_specxy_imag

        low_specxy = torch.complex(low_specxy_R, low_specxy_I)
        # scale for attitude
        scale_attitude = torch.full((low_specxy_real.size(2),), 2 / low_specxy_real.size(2)).to(low_specxy_imag.device)
        scale_attitude[0] = 1 / low_specxy_real.size(2)
        scale_attitude = scale_attitude.unsqueeze(0).unsqueeze(0).repeat(low_specxy_imag.size(0), low_specxy_imag.size(1), 1)

        # 相位角 = arctan(imag/real)  幅值= sqrt(real^2 + imag^2)
        attitude = torch.sqrt(torch.pow(low_specxy_I, 2) + torch.pow(low_specxy_R, 2)) * scale_attitude
        theta = torch.angle(low_specxy)

        output = self.D2V(attitude, theta, tau, self.f, mode)
        return output

    def D2V(self, attitude, theta, tau, f, mode):
        _, D, _ = attitude.shape
        tau = self.mark_fusion(tau.transpose(1,-1)).repeat(1, 1, D) # (B,L+O,d_feature)
        tau = tau.transpose(1, -1)  # (B,d_feature,L+O)
        theta = theta.unsqueeze(2)

        attitude_topk, _ = self.keep_topk(attitude, self.top_k)
        attitude_topk = attitude_topk.unsqueeze(2)
        attitude_topk.detach().requires_grad = False

        attitude = attitude.unsqueeze(2)
        tau = tau.unsqueeze(2).transpose(2, -1)

        w_fourier = self.w_fourier.unsqueeze(-1).to(tau.device)
        w_tau = torch.einsum('bdln,fn->bdlf', tau, w_fourier)
        #v1 = attitude * f(w_tau+theta)
        v1 = attitude * f(w_tau)
        #v1 = f(w_tau)

        vision_input = v1.clone()
        # vision
        if mode == "test" and self.d2v_vision_flag == True:
            self.plot_cos_waves(vision_input[0,0,:,:])
            self.d2v_vision_flag = False

        return v1

    def keep_topk(self, tensor, k):
        """
        保留tensor中前k个最大值,其余值设为0

        参数:
        tensor (torch.Tensor): 输入tensor
        k (int): 保留的最大值个数

        返回:
        torch.Tensor: 保留前k个最大值的tensor
        """
        # 获取tensor中前k个最大值的索引
        _, indices = torch.topk(tensor, k, dim=-1)

        # 创建一个与输入tensor形状相同的全0tensor
        mask = torch.zeros_like(tensor)

        # 将前k个最大值位置设为原tensor对应值,其余位置设为0
        mask.scatter_(-1, indices, 1)

        # 将输入tensor与掩码相乘,保留前k个最大值,其余值变为0
        output = tensor * mask.float()

        return output, indices

    def plot_cos_waves(self, v1):
        """
            在同一张图上绘制多个不同频率和幅值的正弦波形。

            参数:
            time_range (tuple): 时间范围,格式为 (start, end)
            num_points (int): 数据点的数量
            frequencies (list): 正弦波的频率列表,单位为 Hz
            amplitudes (list): 正弦波的幅值列表
            """
        # 1. 计算每个特征（列）的均值
        feature_mean_vals = torch.mean(v1, dim=0)  # 获取每列的均值

        # 2. 获取均值的前 8 个特征的索引
        _, top_k_indices = torch.topk(feature_mean_vals, self.top_k)  # 获取均值最大的 8 个值的索引

        # 3. 提取这 8 个特征的时间序列
        top_k_features = v1[:, top_k_indices]  # 通过索引提取这 8 个特征

        # 创建子图
        fig, axes = plt.subplots(self.top_k, 1, figsize=(10, 8), sharex=True)

        # 绘制每个正弦波
        for j in range(self.top_k):
            y_wave = top_k_features[:, j].detach().cpu()
            axes[j].plot(y_wave)
            axes[j].set_title(f'Feature {top_k_indices[j].item()}')
            axes[j].set_ylabel('振幅')
        # 添加x轴标签
        axes[-1].set_xlabel('时间 (秒)')
        # 调整子图间距并显示
        plt.subplots_adjust(hspace=0.5)
        # plt.show()
        plt.savefig(self.save_path + '_multi_cos_vision.png')
        print(f"Date2Vec图片已保存")
