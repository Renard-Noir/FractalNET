""" Utilities """
import os
import glob
import logging
import shutil
import torch
import torchvision.datasets as dset
import numpy as np
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional


def get_logger(file_path):
    """ Make python logger """
    # [!] Since tensorboardX use default logger (e.g. logging.info()), we should use custom logger
    logger = logging.getLogger('fractal')
    log_format = '%(asctime)s | %(message)s'
    formatter = logging.Formatter(log_format, datefmt='%m/%d %I:%M:%S %p')
    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)

    return logger


def param_size(model):
    """ Compute parameter size in Mega """
    n_params = sum(np.prod(v.size()) for k, v in model.named_parameters())
    return n_params / 1000. / 1000.


def get_module_device(module):
    """ Get pytorch module device """
    return next(module.parameters()).device


class AverageMeter():
    """ Computes and stores the average and current value """
    def __init__(self):
        self.reset()

    def reset(self):
        """ Reset all statistics """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """ Update statistics """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """ Computes the precision@k for the specified values of k """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    # one-hot case
    if target.ndimension() > 1:
        target = target.max(1)[1]

    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(1.0 / batch_size))

    return res


def save_checkpoint(state, ckpt_dir, is_best=False):
    filename = os.path.join(ckpt_dir, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(ckpt_dir, 'best.pth.tar')
        shutil.copyfile(filename, best_filename)


def copy_scripts(src_pattern, dest_dir):
    dest_dir = os.path.join(dest_dir, "scripts")
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    src_paths = glob.glob(src_pattern)
    for path in src_paths:
        dest_path = os.path.join(dest_dir, os.path.basename(path))
        shutil.copyfile(path, dest_path)

def init_metadata(model_id: str, args: argparse.Namespace, classes: List[str]) -> Dict[str, Any]:
    """Создаёт словарь с начальными метаданными для модели."""
    return {
        "timestamp": datetime.now().strftime('%Y%m%d_%H%M%S'),
        "model_id": model_id,
        "base_model": None,
        "optimizer": args.optimizer,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "C": getattr(args, 'C', None),           # для совместимости с параметрами FractalNet
        "channels": getattr(args, 'channels', None),
        "drop_path": getattr(args, 'p_ldrop', None),
        "classes": classes,
        "num_classes": len(classes),
        "start_time": None,
        "end_time": None,
        "last_checkpoint_epoch": None,
        "best": None,
        "history": {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": []
        },
        "final": None,
        "best_summary": None
    }

def load_or_create_metadata(meta_path: Path, model_id: str, args: argparse.Namespace,
                            classes: List[str]) -> (Dict[str, Any], List[Dict[str, Any]]):
    """
    Загружает существующий metadata.json или создаёт новый.
    Возвращает (entry, container), где entry – словарь текущей модели,
    container – весь список записей.
    """
    entry_template = init_metadata(model_id, args, classes)

    if meta_path.exists():
        data = json.loads(meta_path.read_text())
        if isinstance(data, list):
            for entry in data:
                if entry.get("model_id") == model_id:
                    return entry, data
            # не найдено – добавляем
            data.append(entry_template)
            meta_path.write_text(json.dumps(data, indent=2))
            return data[-1], data
        elif isinstance(data, dict):
            if data.get("model_id") == model_id:
                return data, [data]
            else:
                arr = [data, entry_template]
                meta_path.write_text(json.dumps(arr, indent=2))
                return arr[-1], arr
        else:
            # неизвестный формат – перезаписываем
            meta_path.write_text(json.dumps([entry_template], indent=2))
            return entry_template, [entry_template]
    else:
        meta_path.write_text(json.dumps([entry_template], indent=2))
        return entry_template, [entry_template]

def persist_metadata(meta_path: Path, container: List[Dict[str, Any]]) -> None:
    """Сохраняет контейнер метаданных в файл."""
    meta_path.write_text(json.dumps(container, indent=2))

def load_checkpoint(path: Path, model: nn.Module, optimizer: optim.Optimizer = None,
                    device: torch.device = None) -> Dict[str, Any]:
    """
    Загружает чекпоинт. Поддерживает разные форматы.
    Возвращает загруженный словарь (может содержать 'epoch', 'best_val_acc' и др.)
    """
    data = torch.load(path, map_location=device)
    # Попытка загрузить состояние модели
    if isinstance(data, dict):
        if "model_state" in data:
            model.load_state_dict(data["model_state"])
        elif "state_dict" in data:
            model.load_state_dict(data["state_dict"])
        else:
            # возможно, это просто state_dict
            try:
                model.load_state_dict(data)
            except Exception:
                pass
    else:
        model.load_state_dict(data)

    if optimizer is not None and isinstance(data, dict) and "optimizer_state" in data:
        try:
            optimizer.load_state_dict(data["optimizer_state"])
        except Exception:
            pass

    return data

def accuracy_per_class(output: torch.Tensor, target: torch.Tensor, num_classes: int,
                       topk=(1,)) -> (List[float], List[float], List[int], List[int]):
    """
    Возвращает top-1 accuracy для каждого класса, а также общие correct/total.
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        # Инициализируем счётчики по классам
        correct_per_class = [0] * num_classes
        total_per_class = [0] * num_classes

        for i in range(batch_size):
            label = target[i].item()
            total_per_class[label] += 1
            if correct[0, i]:   # top-1 правильный
                correct_per_class[label] += 1

        return correct_per_class, total_per_class
    
def evaluate_with_per_class(model: nn.Module, loader: DataLoader, criterion: nn.Module,
                            device: torch.device, num_classes: int) -> (float, float, List[float]):
    """
    Вычисляет средние потери, общую точность и точность по каждому классу.
    Возвращает (avg_loss, overall_acc, per_class_acc).
    """
    model.eval()
    running_loss = 0.0
    correct_total = 0
    total_samples = 0

    correct_per_class = [0] * num_classes
    total_per_class = [0] * num_classes

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            outputs = model(X)
            loss = criterion(outputs, y)

            running_loss += loss.item() * X.size(0)
            _, preds = outputs.max(1)
            correct_total += preds.eq(y).sum().item()
            total_samples += y.size(0)

            # per-class
            for i in range(y.size(0)):
                label = y[i].item()
                total_per_class[label] += 1
                if preds[i] == label:
                    correct_per_class[label] += 1

    avg_loss = running_loss / total_samples if total_samples > 0 else 0.0
    overall_acc = correct_total / total_samples if total_samples > 0 else 0.0
    per_class_acc = [c / t if t > 0 else 0.0 for c, t in zip(correct_per_class, total_per_class)]

    return avg_loss, overall_acc, per_class_acc