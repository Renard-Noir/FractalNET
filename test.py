"""Evaluation script for FractalNet checkpoints."""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import TestConfig
from datasets import get_split_dataset
from fractal import FractalNet
import utils


config = TestConfig()
device = torch.device(f'cuda:{config.gpu}' if torch.cuda.is_available() else 'cpu')
use_amp = bool(config.amp and torch.cuda.is_available())


def build_model(data_shape):
    model = FractalNet(
        data_shape,
        config.columns,
        config.init_channels,
        p_ldrop=0.0,
        dropout_probs=[0.0] * config.blocks,
        gdrop_ratio=0.0,
        gap=config.gap,
        init='torch',
        pad_type=config.pad,
        doubling=config.doubling,
        consist_gdrop=True,
        dropout_pos=config.dropout_pos,
    )
    if config.channels_last and torch.cuda.is_available():
        model = model.to(memory_format=torch.channels_last)
    return model.to(device)


def main():
    criterion = nn.CrossEntropyLoss().to(device)
    dataset, data_shape = get_split_dataset(config.data, config.data_path, config.split, config.img_size)
    loader_kwargs = {
        'batch_size': config.batch_size,
        'shuffle': False,
        'num_workers': config.workers,
        'pin_memory': torch.cuda.is_available(),
    }
    if config.workers > 0:
        loader_kwargs['persistent_workers'] = True
        loader_kwargs['prefetch_factor'] = 2
    loader = DataLoader(dataset, **loader_kwargs)
    model = build_model(data_shape)

    ckpt_path = Path(config.path) / config.model_file
    if not ckpt_path.exists():
        raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')

    utils.load_checkpoint(ckpt_path, model, device=device)
    metrics = utils.evaluate_model(
        model, loader, criterion, device,
        deepest=config.deepest,
        return_predictions=True,
        use_amp=use_amp,
        channels_last=config.channels_last and torch.cuda.is_available(),
    )

    print(f'Split: {config.split}')
    print(f'Accuracy: {metrics["accuracy"]:.4%}')
    print(f'Precision (macro): {metrics["precision_macro"]:.4f}')
    print(f'Recall (macro): {metrics["recall_macro"]:.4f}')
    print(f'F1 (macro): {metrics["f1_macro"]:.4f}')
    print(f'Loss: {metrics["loss"]:.4f}')


if __name__ == '__main__':
    main()
