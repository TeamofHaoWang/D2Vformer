import argparse
import os
from time import strftime
from typing import Callable
from time import strftime
import os

import shutil

#TODO 创建输出目录的函数
def create_output_dir(args):
    if args.output_dir==None:
        #  存储对应的超参数
        if args.save_log:# 进入这个部分的代码

            args.output_dir = os.path.join('experiments')
            # 找到最大的 exp 对应的下标且加一
            current_exp = 0
            if os.path.exists(args.output_dir):
                exp_values = [int(f[3:]) for f in os.listdir(args.output_dir) if f.startswith('exp')]
                current_exp = max(exp_values) + 1 if exp_values else 0

            if args.exp_num != -1 and args.exp_num < current_exp:
                current_exp = args.exp_num

            args.output_dir = os.path.join(args.output_dir, 'exp{}'.format(current_exp))
    else:
        if not os.path.exists(args.output_dir):
            print('路径为{0}的输出路径不存在'.format(args.output_dir))
        else:
            shutil.rmtree(args.output_dir) # 删除对应的文件夹下所有的文件

    current_time = strftime('%Y-%m-%d_%H-%M-%S')
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    with open(os.path.join(args.output_dir, '{}.time'.format(current_time)), 'a+') as f:
        pass

    with open(os.path.join(args.output_dir, 'README'), 'a+') as f:
        f.write(args.desc)

    return args