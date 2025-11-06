import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding, DataEmbedding_value
from layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp
import numpy as np
import matplotlib.pyplot as plt
from layers.Date2Vec import Date2Vec_Fourier




class D2V_Autoformer(nn.Module):
    """
    Autoformer is the first method to achieve the series-wise connection,
    with inherent O(LlogL) complexity
    """
    def __init__(self, configs):
        super(D2V_Autoformer, self).__init__()
        #Time dimension of the encoder input using the timestamp length of the historical time series
        self.seq_len = configs.seq_len
        #Timestamp length of the historical time series entered by the decoder。
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.output_attention = False

        # Decomp，Kernel size of the incoming parameter mean filter
        kernel_size = configs.moving_avg
        self.decomp = series_decomp(kernel_size)

        # Embedding
        # The embedding operation, since time series are naturally sequential in timing, the role of embedding here is more to adjust the dimensionality
        self.enc_embedding = DataEmbedding_value(configs.d_feature, configs.d_model, configs.embed, configs.freq,
                                                  configs.dropout)
        self.dec_embedding = DataEmbedding_value(configs.d_feature, configs.d_model, configs.embed, configs.freq,
                                                  configs.dropout)

        # D2V Embedding
        self.D2V_position = Date2Vec_Fourier(self.seq_len, configs.T2V_outmodel, self.d_feature,
                                         d_mark=len(self.mark_index), save_path=configs.output_path)


        # Encoder，Multi-coded layer stacking is used
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=False),
                        configs.d_model, configs.n_heads),
                    #Feature dimension setting during encoding
                    configs.d_model,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    #activation function
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            #Time series are usually applied using the Layernorm and not the BN layer
            norm_layer=my_Layernorm(configs.d_model)
        )
        # Decoder is also stacked with multiple decoders.
        self.decoder = Decoder(
            [
                DecoderLayer(
                    # As in the traditional Transformer structure, the first attention of the decoder needs to be masked to ensure that the prediction at the current position cannot see the previous content.
                    #This approach is derived from NLP practice, but in the case of temporal prediction, there should be no need to use the mask mechanism.
                    #As you can see in the subsequent code, none of the attention modules here actually use the mask mechanism.
                    AutoCorrelationLayer(
                        AutoCorrelation(True, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=False),
                        configs.d_model, configs.n_heads),
                    AutoCorrelationLayer(
                        AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
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

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,**kwargs):
        mode = kwargs['mode']
        # decomp init
        # Because generative prediction needs to be used, it is necessary to occupy the prediction section with means and zeros.
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]], device=x_enc.device)
        seasonal_init, trend_init = self.decomp(x_enc)
        # decoder input
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = torch.cat([seasonal_init[:, -self.label_len:, :], zeros], dim=1)
        # D2V Embedding
        # mark concat
        D2V_input_mark = torch.cat([x_mark_enc, x_mark_dec], dim=1)
        D2V_input_mark = D2V_input_mark[:, :, 1:3]

        D2V_output = self.D2V_position(x_enc.permute(0,2,1),D2V_input_mark.permute(0,2,1), mode)  # x: [Batch, Channel, Input length]


        D2V_output = torch.sum(D2V_output,dim=-1).permute(0,2,1)  #D2V_output[B,L of x mark enc + L of x mark dec,D]
        x_enc = x_enc + D2V_output[:,:x_enc.shape[1],:]

        # enc
        enc_out = self.enc_embedding(x_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        # dec
        seasonal_init = seasonal_init + D2V_output[:,-seasonal_init.shape[1]:,:]
        dec_out = self.dec_embedding(seasonal_init)
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None,
                                                 trend=trend_init)
        # final
        dec_out = trend_part + seasonal_part

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]