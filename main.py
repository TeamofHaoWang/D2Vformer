# -*- coding: utf-8 -*-
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.chdir('')
import io
import sys
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')
from exp.exp import EXP
from utils.setseed import set_seed


if __name__ == '__main__':

    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v == 'True':
            return True
        if v == 'False':
            return False


    import argparse

    parser = argparse.ArgumentParser(description=__doc__)

    # 有关实验模型的参数
    parser.add_argument('--model_name', default='D2Vformer', type=str, help='[D2Vformer]')
    # patch
    parser.add_argument('--patch_len', type=int, default=4, help='patch长度')
    parser.add_argument('--stride', type=int, default=2, help='每一个patch的间隔')
    parser.add_argument('--d_model', default=510,type=int, help='feature dim in model')
    # D2V
    parser.add_argument('--T2V_outmodel', type=int, default=64, help='T2V_outmodel')

    parser.add_argument('--data_name', default='illness', type=str, help='[data:ETTh1,electricity,exchange,ETTm1,traffic]')
    parser.add_argument('--seq_len', default=36, type=int, help='input sequence len')
    parser.add_argument('--label_len', default=24, type=int, help='transfomer decoder input part')
    parser.add_argument('--pred_len', default=12, type=int,help='prediction len,真正要预测的结果放在list中间')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')

    parser.add_argument('--train', default=True, type=str2bool, help='if train')
    parser.add_argument('--resume', default=False, type=str2bool, help='resume from checkpoint')

    parser.add_argument('--loss', default='quantile', type=str, help='quantile,normal')
    parser.add_argument('--seed', default=1, type=int, help='random seed')
    # 调用不同的实验，有些模型需要特殊的实验操作
    parser.add_argument('--exp', default='deep_learning', type=str, help='[deep_learning]')

    # 实验数据相关参数
    parser.add_argument('--d_mark', default=27, type=int, help='date embed dim')
    parser.add_argument('--d_feature', default=7, type=int,
                        help='input data feature dim without date :[Etth1:7 , electricity:321,exchange:8,ETTm1:7,illness:7,traffic:863]')
    parser.add_argument('--c_out', type=int, default=7, help='output size')

    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')

    # 数据增强的操作
    parser.add_argument('--is_shuffle', default=False, type=str2bool)
    parser.add_argument('--is_augmentation', default=False, type=str2bool, help='augmentation for trainset')
    parser.add_argument('--augmentation_s', default=10, type=int, help='augmentation stride')

    # 超参

    parser.add_argument('--d_ff', default=1024, type=int, help='feature dim2 in model')
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
    parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--lr_d', default=0.05, type=float, help='initial learning rate for discriminator of gan')

    parser.add_argument('--epoches', default=200, type=int, help='Train Epoches')
    parser.add_argument('--patience', default=5, type=int, help='Early Stop patience')

    parser.add_argument('--quantiles', default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], type=int,
                        help='分位数损失的ρ参数')

    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, '
                             'b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')

    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=2, help='num of decoder layers')
    parser.add_argument('--n_heads', type=int, default=3, help='num of heads')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')

    parser.add_argument('--factor', type=int, default=5)


    parser.add_argument('--patch_stride', type=int, default=8)

    parser.add_argument('--fourier_decomp_ratio', type=float, default=0.5, help='频率拆解模块参数设置')
    parser.add_argument('--stronger', type=int, default=2, help='Top-K Fourier bases')



    # Nbeats
    parser.add_argument('--trend_power', default=2, type=int, help='max power for trend')
    parser.add_argument('--n_stacks', default=2, type=int, help='number of Trend stack')

    # FEB 降噪
    parser.add_argument('--modes', type=int, default=32)

    parser.add_argument('--weight_decay', default=0.001, type=float)

    # info
    parser.add_argument('--info', type=str, default='None', help='extra information')

    args = parser.parse_args()
    # set_seed(args.seed)
    print(f"|{'=' * 101}|")
    # 使用__dict__方法获取参数字典，之后遍历字典
    for key, value in args.__dict__.items():
        # 因为参数不一定都是str，需要将所有的数据转成str
        print(f"|{str(key):>50s}|{str(value):<50s}|")
    print(f"|{'=' * 101}|")

    if args.exp == 'deep_learning':
        if args.is_augmentation:
            exp = EXP_augmentation(args)
            if args.train:
                exp.train()
            exp.test()
        else:
            exp = EXP(args)
            if args.train:
                exp.train()
            exp.test()

