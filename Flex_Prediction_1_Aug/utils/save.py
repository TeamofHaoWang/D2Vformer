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

    # '''保留checkpoint'''
    # def save_ckpt(self, checkpoint_stats, backbone_stats, epoch, cur_metric, point_round=5):
    #     # save checkpoint
    #     if self.check():
    #         return
    #
    #     checkpoint_name = '{}_epoch_{}_{}_{}_checkpoint.ckpt'.format(
    #         self.model_name, epoch, self.metric_name, round(cur_metric, point_round))
    #     checkpoint_path = os.path.join(self.output_dir, checkpoint_name)
    #
    #     self.ckpt_save_cnt += 1
    #     if checkpoint_stats is not None and self.ckpt_save_cnt >= self.ckpt_save_freq:
    #         checkpoint_stats['save_cnt'] = self.ckpt_save_cnt
    #         self.ckpt_save_cnt = 0
    #         torch.save(checkpoint_stats, checkpoint_path)
    #
    #         latest_ckpts = [f for f in os.listdir(self.output_dir) if f.startswith('latest')]
    #         if latest_ckpts:
    #             latest_ckpt_name = latest_ckpts[0]
    #             latest_epoch = self.parse_metric(latest_ckpt_name, 'latest')
    #             if epoch > latest_epoch:
    #                 os.remove(os.path.join(self.output_dir, latest_ckpt_name))
    #         latest_ckpt_path = os.path.join(self.output_dir, 'latest_{}_checkpoint.ckpt'.format(epoch))
    #         torch.save(checkpoint_stats, latest_ckpt_path)
    #
    #     # save the best
    #     if self._compare(cur_metric, self.last_metric):
    #         self.last_metric = cur_metric
    #         get_best = True
    #
    #         current_best_models = [f for f in os.listdir(self.output_dir) if f.startswith('_best')]
    #
    #         assert len(current_best_models) <= 1
    #
    #         if len(current_best_models) == 1:
    #             current_best_model_name = current_best_models[0]
    #
    #             last_best_metric = self.parse_metric(current_best_model_name, self.metric_name)
    #
    #             if self._compare(last_best_metric, self.last_metric):
    #                 self.last_metric = last_best_metric
    #                 get_best = False
    #             else:
    #                 current_best_model_path = os.path.join(self.output_dir, current_best_model_name)
    #                 os.remove(current_best_model_path)
    #
    #         if get_best and backbone_stats is not None:
    #             best_model_name = checkpoint_name.replace('checkpoint', 'model')
    #             best_model_name = '_best_' + best_model_name
    #             best_model_path = os.path.join(self.output_dir, best_model_name)
    #             torch.save(backbone_stats, best_model_path)
    #
    #             simple_name = 'best_{}.pth'.format(self.model_name)
    #             simple_path = os.path.join(self.output_dir, simple_name)
    #             torch.save(backbone_stats, simple_path)


def ddp_module_replace(param_ckpt):
    return {k.replace('module.', ''): v.cpu() for k, v in param_ckpt.items()}


# def resume_from_checkpoint(path, backbone, head, optimizer, lr_optimizer, ddp=False):
#     checkpoint = torch.load(path, map_location='cpu')
#
#     backbone_checkpoint = checkpoint['backbone'] if ddp else ddp_module_replace(checkpoint['backbone'])
#     head_checkpoint = checkpoint['head'] if ddp else ddp_module_replace(checkpoint['head'])
#
#     backbone.load_state_dict(backbone_checkpoint)
#     head.load_state_dict(head_checkpoint)
#     optimizer.load_state_dict(checkpoint['optimizer'])
#     lr_optimizer.load_state_dict(checkpoint['lr_optimizer'])
#     start_epoch = checkpoint['epoch'] + 1
#
#     return backbone, head, optimizer, lr_optimizer, start_epoch
