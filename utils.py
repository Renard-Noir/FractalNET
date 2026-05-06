"""Utility helpers for training, evaluation and plots."""
from __future__ import annotations

import glob
import json
import logging
import os
import random
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


PLOT_DPI = 150


def get_logger(file_path: str):
    logger = logging.getLogger(f'fractal.{file_path}')
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        return logger

    formatter = logging.Formatter('%(asctime)s | %(message)s', datefmt='%m/%d %I:%M:%S %p')
    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def ensure_dir(path: str | Path):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(path: str | Path, data: Any):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path: str | Path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = float(val)
        self.sum += float(val) * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)


def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    if target.ndimension() > 1:
        target = target.max(1)[1]
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(1.0 / batch_size))
    return res


def param_size(model):
    n_params = sum(np.prod(v.size()) for _, v in model.named_parameters())
    return n_params / 1_000_000.0


def save_checkpoint(state: Dict[str, Any], ckpt_dir: str | Path, is_best=False):
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    filename = ckpt_dir / 'checkpoint.pth.tar'
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, ckpt_dir / 'best.pth.tar')


def copy_scripts(src_pattern: str, dest_dir: str | Path):
    dest_dir = Path(dest_dir) / 'scripts'
    dest_dir.mkdir(parents=True, exist_ok=True)
    for path in glob.glob(src_pattern):
        src = Path(path)
        if src.is_file():
            shutil.copyfile(src, dest_dir / src.name)


def load_checkpoint(path: str | Path, model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer] = None,
                    device: Optional[torch.device] = None):
    data = torch.load(path, map_location=device)
    if isinstance(data, dict) and 'model_state' in data:
        model.load_state_dict(data['model_state'])
    elif isinstance(data, dict) and 'state_dict' in data:
        model.load_state_dict(data['state_dict'])
    else:
        model.load_state_dict(data)
    if optimizer is not None and isinstance(data, dict) and 'optimizer_state' in data:
        optimizer.load_state_dict(data['optimizer_state'])
    return data


def load_pretrained_weights(model: torch.nn.Module, checkpoint_path: str | Path, device: torch.device):
    data = torch.load(checkpoint_path, map_location=device)
    state = data.get('model_state', data.get('state_dict', data))
    model_state = model.state_dict()
    filtered = {}
    skipped = []
    for k, v in state.items():
        if k in model_state and model_state[k].shape == v.shape:
            filtered[k] = v
        else:
            skipped.append(k)
    model_state.update(filtered)
    model.load_state_dict(model_state)
    return {'loaded_keys': sorted(filtered.keys()), 'skipped_keys': skipped}


def evaluate_model(model: torch.nn.Module, loader, criterion, device: torch.device, deepest: bool = False,
                   return_predictions: bool = False, use_amp: bool = False, channels_last: bool = False):
    model.eval()
    losses = AverageMeter()
    top1 = AverageMeter()
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for x, y in loader:
            if channels_last and x.ndim == 4 and torch.cuda.is_available():
                x = x.contiguous(memory_format=torch.channels_last)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp and torch.cuda.is_available()):
                logits = model(x, deepest=deepest)
                loss = criterion(logits, y)
            prec1 = accuracy(logits, y, topk=(1,))[0]
            losses.update(loss.item(), x.size(0))
            top1.update(prec1.item(), x.size(0))
            if return_predictions:
                all_targets.extend(y.cpu().tolist())
                all_preds.extend(torch.argmax(logits, dim=1).cpu().tolist())

    result = {
        'loss': losses.avg,
        'accuracy': top1.avg,
    }
    if return_predictions:
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets, all_preds, average='macro', zero_division=0
        )
        cm = confusion_matrix(all_targets, all_preds)
        result.update({
            'targets': all_targets,
            'predictions': all_preds,
            'precision_macro': float(precision),
            'recall_macro': float(recall),
            'f1_macro': float(f1),
            'confusion_matrix': cm.tolist(),
        })
    return result


def plot_training_curves(history: Dict[str, List[float]], save_path: str | Path):
    epochs = range(1, len(history.get('train_loss', [])) + 1)
    if not list(epochs):
        return
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(list(epochs), history.get('train_loss', []), label='train_loss')
    plt.plot(list(epochs), history.get('val_loss', []), label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss curves')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(list(epochs), history.get('train_acc', []), label='train_acc')
    plt.plot(list(epochs), history.get('val_acc', []), label='val_acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy curves')
    plt.legend()

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=PLOT_DPI, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(cm: np.ndarray, save_path: str | Path, title: str = 'Confusion matrix'):
    plt.figure(figsize=(12, 10))
    plt.imshow(cm, interpolation='nearest')
    plt.title(title)
    plt.colorbar()
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=PLOT_DPI, bbox_inches='tight')
    plt.close()


def plot_final_metrics(metrics: Dict[str, float], save_path: str | Path):
    keys = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
    values = [float(metrics.get(k, 0.0)) for k in keys]
    plt.figure(figsize=(7, 4))
    plt.bar(keys, values)
    plt.ylim(0.0, 1.0)
    plt.ylabel('Score')
    plt.title('Final evaluation metrics')
    for idx, value in enumerate(values):
        plt.text(idx, min(value + 0.02, 0.98), f'{value:.3f}', ha='center')
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=PLOT_DPI, bbox_inches='tight')
    plt.close()
