"""Configuration helpers for train/test scripts."""
from __future__ import annotations

import argparse
import os
import shutil


class BaseConfig(argparse.Namespace):
    def print_params(self, prtf=print):
        prtf("\nParameters:")
        for attr, value in sorted(vars(self).items()):
            prtf(f"{attr.upper()}={value}")
        prtf("")

    def as_markdown(self) -> str:
        text = "|name|value|\n|-|-|\n"
        for attr, value in sorted(vars(self).items()):
            text += f"|{attr}|{value}|\n"
        return text


def _common_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("FractalNet config")
    parser.add_argument('--name', required=True)
    parser.add_argument('--data', default='stanford_dogs', help='stanford_dogs / cifar10 / cifar100')
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--img_size', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--print_freq', type=int, default=50)
    parser.add_argument('--overwrite', action='store_true', default=False)
    return parser


class Config(BaseConfig):
    def build_parser(self):
        parser = _common_parser()
        parser.add_argument('--lr', type=float, default=1e-3)
        parser.add_argument('--momentum', type=float, default=0.9)
        parser.add_argument('--epochs', type=int, default=20)
        parser.add_argument('--optimizer', type=str, default='adam', choices=['sgd', 'adam', 'adasmoothdelta'])
        parser.add_argument('--resume', type=str, default=None)
        parser.add_argument('--pretrained_checkpoint', type=str, default=None)
        parser.add_argument('--save_interval', type=int, default=0)
        parser.add_argument('--test_after_train', action='store_true', default=False)
        parser.add_argument('--aug_lv', type=int, default=1,
                            help='0: no aug, 1: crop+flip, 2: crop+flip+cutout')
        parser.add_argument('--mixup_alpha', type=float, default=0.2)
        parser.add_argument('--init_channels', type=int, default=64)
        parser.add_argument('--gdrop_ratio', type=float, default=0.5)
        parser.add_argument('--p_ldrop', type=float, default=0.15)
        parser.add_argument('--dropout_probs', default='0.0,0.1,0.2,0.3,0.4')
        parser.add_argument('--blocks', type=int, default=5)
        parser.add_argument('--columns', type=int, default=3)
        parser.add_argument('--off-drops', action='store_true', default=False)
        parser.add_argument('--gap', type=int, default=1)
        parser.add_argument('--init', default='torch', choices=['xavier', 'he', 'torch'])
        parser.add_argument('--pad', default='reflect', choices=['zero', 'reflect'])
        parser.add_argument('--doubling', action='store_true', default=False)
        parser.add_argument('--gdrop_type', default='ps-consist', choices=['ps', 'ps-consist'])
        parser.add_argument('--dropout_pos', default='CDBR', choices=['CDBR', 'CBRD', 'FD'])
        parser.add_argument('--amp', action='store_true', default=False,
                            help='enable automatic mixed precision on CUDA')
        parser.add_argument('--channels_last', action='store_true', default=False,
                            help='use channels_last memory format on CUDA')
        parser.add_argument('--compile_model', action='store_true', default=False,
                            help='use torch.compile when available')
        parser.add_argument('--eval_every', type=int, default=1,
                            help='run validation once every N epochs')
        parser.add_argument('--freeze_head_epochs', type=int, default=0,
                            help='for finetuning: train only classifier head for the first N epochs')
        parser.add_argument('--prefetch_factor', type=int, default=2,
                            help='DataLoader prefetch factor when workers > 0')
        return parser

    def __init__(self):
        parser = self.build_parser()
        args = parser.parse_args()
        super().__init__(**vars(args))

        self.data = self.data.lower().strip()
        self.path = os.path.join('./runs', self.name)
        self.dropout_probs = [float(p.strip()) for p in self.dropout_probs.split(',') if p.strip()]
        if len(self.dropout_probs) != self.blocks:
            raise ValueError('dropout_probs length must equal blocks')
        self.consist_gdrop = self.gdrop_type == 'ps-consist'
        self.lr_milestone = self._build_lr_milestones(self.epochs)
        self.eval_every = max(1, int(self.eval_every))
        self.prefetch_factor = max(1, int(self.prefetch_factor))
        self.freeze_head_epochs = max(0, int(self.freeze_head_epochs))

        if self.off_drops:
            self.dropout_probs = [0.0] * self.blocks
            self.p_ldrop = 0.0
            self.gdrop_ratio = 0.0

        if os.path.exists(self.path) and self.overwrite and not self.resume:
            shutil.rmtree(self.path)

    @staticmethod
    def _build_lr_milestones(epochs: int):
        left = max(epochs // 2, 1)
        out = [left]
        for _ in range(3):
            left = max(left // 2, 1)
            out.append(out[-1] + left)
        return sorted(set(x for x in out if x < epochs))


class TestConfig(BaseConfig):
    def build_parser(self):
        parser = _common_parser()
        parser.add_argument('--init_channels', type=int, default=64)
        parser.add_argument('--blocks', type=int, default=5)
        parser.add_argument('--columns', type=int, default=3)
        parser.add_argument('--gap', type=int, default=1)
        parser.add_argument('--pad', default='reflect', choices=['zero', 'reflect'])
        parser.add_argument('--doubling', action='store_true', default=False)
        parser.add_argument('--dropout_pos', default='CDBR', choices=['CDBR', 'CBRD', 'FD'])
        parser.add_argument('--split', default='test', choices=['evaluate', 'test', 'valid'])
        parser.add_argument('--model_file', default='best.pth.tar')
        parser.add_argument('--deepest', action='store_true', default=False)
        parser.add_argument('--amp', action='store_true', default=False,
                            help='enable mixed precision for evaluation on CUDA')
        parser.add_argument('--channels_last', action='store_true', default=False)
        return parser

    def __init__(self):
        parser = self.build_parser()
        args = parser.parse_args()
        super().__init__(**vars(args))
        self.data = self.data.lower().strip()
        if self.split == 'valid':
            self.split = 'evaluate'
        self.path = os.path.join('./runs', self.name)
