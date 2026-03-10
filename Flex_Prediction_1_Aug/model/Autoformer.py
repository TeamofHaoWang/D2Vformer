import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding, DataEmbedding_wo_pos
from layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp


class Autoformer(nn.Module):
    """
    Autoformer is the first method to achieve the series-wise connection,
    with inherent O(LlogL) complexity
    """
    def __init__(self, configs):
        super(Autoformer, self).__init__()
        #利用历史时间序列的时间戳长度，编码器输入的时间维度
        self.seq_len = configs.seq_len
        #解码器输入的历史时间序列的时间戳长度。
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.output_attention = False
        self.d_model = configs.d_model

        # Decomp，传入参数均值滤波器的核大小
        kernel_size = configs.moving_avg
        self.decomp = series_decomp(kernel_size)

        # Embedding
        # embedding操作，由于时间序列天然在时序上具有先后关系，因此这里embedding的作用更多的是为了调整维度
        self.enc_embedding = DataEmbedding_wo_pos(configs.d_feature, configs.d_model, configs.embed, configs.freq,
                                                  configs.dropout)
        self.dec_embedding = DataEmbedding_wo_pos(configs.d_feature, configs.d_model, configs.embed, configs.freq,
                                                  configs.dropout)
        self.linear = nn.Linear(configs.d_feature,configs.d_model)


        # Encoder，采用的是多编码层堆叠
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        #这里的第一个False表明是否使用mask机制。
                        AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=False),
                        configs.d_model, configs.n_heads),
                    #编码过程中的特征维度设置
                    configs.d_model,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    #激活函数
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            #时间序列通常采用Layernorm而不适用BN层
            norm_layer=my_Layernorm(configs.d_model)
        )
        # Decoder也是才是用多解码器堆叠
        self.decoder = Decoder(
            [
                DecoderLayer(
                    #如同传统的Transformer结构，decoder的第一个attention需要mask，保证当前的位置的预测不能看到之前的内容
                    #这个做法是来源于NLP中的作法，但是换成时序预测，实际上应该是不需要使用mask机制的。
                    #而在后续的代码中可以看出，这里的attention模块实际上都没有使用mask机制。

                    #self-attention，输入全部来自于decoder自身
                    AutoCorrelationLayer(
                        AutoCorrelation(True, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=False),
                        configs.d_model, configs.n_heads),
                    #cross-attention，输入一部分来自于decoder，另一部分来自于encoder的输出
                    AutoCorrelationLayer(
                        AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    #任务要求的输出特征维度
                    configs.c_out,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # decomp init
        # 因为需要使用生成式预测，所以需要用均值和0来占位，占住预测部分的位置。
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        zeros = torch.zeros([x_enc.shape[0], self.pred_len, x_enc.shape[2]], device=x_enc.device)
        seasonal_init, trend_init = self.decomp(x_enc)
        # decoder input
        trend_init = torch.cat([trend_init[:, -self.pred_len:, :], mean], dim=1)
        seasonal_init = torch.cat([seasonal_init[:, -self.pred_len:, :], zeros], dim=1)
        # enc
        # enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out = self.linear(x_enc.view(x_enc.size(0)*x_enc.size(1),x_enc.size(2)))
        enc_out = enc_out.view(x_enc.size(0),x_enc.size(1),self.d_model)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        # dec
        # dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        dec_out = self.linear(seasonal_init.view(seasonal_init.size(0)*seasonal_init.size(1),seasonal_init.size(2)))
        dec_out = dec_out.view(seasonal_init.size(0),seasonal_init.size(1),self.d_model)
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask,
                                                 trend=trend_init)
        # final
        dec_out = trend_part + seasonal_part

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]