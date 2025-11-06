import torch

import os
import yaml
import json
from utils.model_tool import get_item


class SaveManager(object):
    def __init__(self, output_dir, model_name, metric_name, ckpt_save_freq=1, compare_type='gt', last_metric=None):
        self.output_dir = output_dir
        self.last_metric = last_metric
        self.model_name = model_name
        self.metric_name = metric_name
        self.ckpt_save_freq = ckpt_save_freq
        self.compare_type = compare_type

        assert ckpt_save_freq > 0
        self.ckpt_save_cnt = ckpt_save_freq - 1

        if compare_type == 'gt':
            if last_metric is None:
                self.last_metric = float('-inf')
        elif compare_type == 'lt':
            if last_metric is None:
                self.last_metric = float('inf')
        else:
            raise ValueError('compare type error!')

        assert len(self.metric_name.split('_')) <= 1, 'metric_name should not use _ to split words'

        self.current_best_models = [f for f in os.listdir(
            self.output_dir) if f.startswith('best')]



    def _compare(self, src, dst):
        if self.compare_type == 'gt':
            return src > dst
        elif self.compare_type == 'lt':
            return src < dst

    def save_epoch_log(self, run_type: str, **kwargs):

        for k, v in kwargs.items():
            kwargs[k] = get_item(v)

        with open(os.path.join(self.output_dir, '{}_epoch_log.txt'.format(run_type)), 'a+') as f:
            f.write(json.dumps(kwargs) + '\n')

    def save_step_log(self, run_type: str, **kwargs):


        for k, v in kwargs.items():
            kwargs[k] = get_item(v)

        with open(os.path.join(self.output_dir, '{}_step_log.txt'.format(run_type)), 'a+') as f:
            f.write(json.dumps(kwargs) + '\n')

    def save_hparam(self, args):
        # args: args from argparse return

        value2save = {k: v for k, v in vars(args).items() if not k.startswith('__') and not k.endswith('__')}
        with open(os.path.join(self.output_dir, 'hparam.yaml'), 'a+') as f:
            f.write(yaml.dump(value2save))

    @staticmethod
    def parse_metric(file_name, metric_name):
        _tmp_str = str(metric_name) + '_'
        idx = file_name.find(_tmp_str) + len(_tmp_str)
        value = float(file_name[idx:file_name.find('_', idx)])
        return value




def ddp_module_replace(param_ckpt):
    return {k.replace('module.', ''): v.cpu() for k, v in param_ckpt.items()}
