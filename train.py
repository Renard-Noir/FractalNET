"""Training script for FractalNet with automatic plots and speed-up options."""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    class SummaryWriter:
        def __init__(self, *args, **kwargs):
            pass
        def add_text(self, *args, **kwargs):
            pass
        def add_scalar(self, *args, **kwargs):
            pass
        def close(self):
            pass

from config import Config
from datasets import get_dataset, get_split_dataset
from fractal import FractalNet
from optimizers import AdaSmoothDelta
import utils


config = Config()
device = torch.device(f'cuda:{config.gpu}' if torch.cuda.is_available() else 'cpu')
use_amp = bool(config.amp and torch.cuda.is_available())
run_path = Path(config.path)
plots_dir = run_path / 'plots'
utils.ensure_dir(run_path)
utils.ensure_dir(plots_dir)

writer = SummaryWriter(log_dir=str(run_path / 'tb'))
logger = utils.get_logger(str(run_path / f'{config.name}.log'))
writer.add_text('config', config.as_markdown(), 0)
config.print_params(logger.info)
utils.copy_scripts('*.py', run_path)


def mixup_data(x, y, alpha=0.2):
    if alpha <= 0.0:
        return x, y, y, 1.0
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    index = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1.0 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1.0 - lam) * criterion(pred, y_b)


def build_model(data_shape):
    model = FractalNet(
        data_shape,
        config.columns,
        config.init_channels,
        p_ldrop=config.p_ldrop,
        dropout_probs=config.dropout_probs,
        gdrop_ratio=config.gdrop_ratio,
        gap=config.gap,
        init=config.init,
        pad_type=config.pad,
        doubling=config.doubling,
        consist_gdrop=config.consist_gdrop,
        dropout_pos=config.dropout_pos,
    )
    if config.channels_last and torch.cuda.is_available():
        model = model.to(memory_format=torch.channels_last)
    model = model.to(device)
    if config.compile_model and hasattr(torch, 'compile'):
        model = torch.compile(model)
    return model


def build_optimizer(model):
    if config.optimizer == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=config.lr, momentum=config.momentum)
    if config.optimizer == 'adam':
        return torch.optim.Adam(model.parameters(), lr=config.lr)
    if config.optimizer == 'adasmoothdelta':
        return AdaSmoothDelta(model.parameters(), lr=config.lr)
    raise ValueError(config.optimizer)


def build_loader(dataset, shuffle: bool):
    kwargs = {
        'batch_size': config.batch_size,
        'shuffle': shuffle,
        'num_workers': config.workers,
        'pin_memory': torch.cuda.is_available(),
    }
    if config.workers > 0:
        kwargs['persistent_workers'] = True
        kwargs['prefetch_factor'] = config.prefetch_factor
    return DataLoader(dataset, **kwargs)


def set_head_only_training(model: nn.Module, head_only: bool):
    base = model.module if hasattr(model, 'module') else model
    trainable_ids = set()
    for layer in reversed(base.layers):
        params = list(layer.parameters(recurse=True))
        if params:
            trainable_ids = {id(p) for p in params}
            break
    if not trainable_ids:
        return
    for p in base.parameters():
        p.requires_grad = (id(p) in trainable_ids) if head_only else True


def to_device(x, y):
    if config.channels_last and x.ndim == 4 and torch.cuda.is_available():
        x = x.contiguous(memory_format=torch.channels_last)
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


def train_one_epoch(loader, model, optimizer, criterion, epoch, scaler):
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    model.train()

    for step, (x, y) in enumerate(loader):
        x, y = to_device(x, y)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
            if config.mixup_alpha > 0:
                mixed_x, y_a, y_b, lam = mixup_data(x, y, config.mixup_alpha)
                logits = model(mixed_x)
                loss = mixup_criterion(criterion, logits, y_a, y_b, lam)
                train_target_for_acc = y
            else:
                logits = model(x)
                loss = criterion(logits, y)
                train_target_for_acc = y

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        prec1 = utils.accuracy(logits, train_target_for_acc, topk=(1,))[0]
        losses.update(loss.item(), x.size(0))
        top1.update(prec1.item(), x.size(0))

        if step % config.print_freq == 0 or step == len(loader) - 1:
            logger.info(
                f'Epoch [{epoch+1}/{config.epochs}] Step [{step+1}/{len(loader)}] '
                f'Loss {losses.avg:.4f} Acc {top1.avg:.4%}'
            )
    return losses.avg, top1.avg


def main():
    logger.info(f'Using device: {device}')
    logger.info(f'AMP enabled: {use_amp}')
    utils.seed_everything(config.seed)
    if torch.cuda.is_available():
        torch.cuda.set_device(config.gpu)
        torch.backends.cudnn.benchmark = True
        if hasattr(torch, 'set_float32_matmul_precision'):
            torch.set_float32_matmul_precision('high')

    train_data, valid_data, data_shape = get_dataset(config.data, config.data_path, config.aug_lv, config.img_size)
    train_loader = build_loader(train_data, shuffle=True)
    valid_loader = build_loader(valid_data, shuffle=False)

    classes = getattr(train_data, 'classes', [str(i) for i in range(data_shape[-1])])
    model = build_model(data_shape)
    logger.info(f'Model params: {utils.param_size(model):.3f} M')

    if config.pretrained_checkpoint:
        loaded = utils.load_pretrained_weights(model, config.pretrained_checkpoint, device)
        logger.info(
            f'Loaded pretrained weights from {config.pretrained_checkpoint}: '
            f'{len(loaded["loaded_keys"])} keys, skipped {len(loaded["skipped_keys"])} keys'
        )

    optimizer = build_optimizer(model)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.lr_milestone, gamma=0.1)
    criterion = nn.CrossEntropyLoss().to(device)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp) if use_amp else None

    start_epoch = 0
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    if config.resume:
        ckpt = utils.load_checkpoint(config.resume, model, optimizer, device)
        start_epoch = int(ckpt.get('epoch', 0))
        best_val_acc = float(ckpt.get('best_val_acc', 0.0))
        history = ckpt.get('history', history)
        if scaler is not None and isinstance(ckpt, dict) and 'scaler_state' in ckpt:
            scaler.load_state_dict(ckpt['scaler_state'])
        logger.info(f'Resumed from {config.resume} at epoch {start_epoch}')

    for epoch in range(start_epoch, config.epochs):
        head_only = bool(config.pretrained_checkpoint and epoch < config.freeze_head_epochs)
        set_head_only_training(model, head_only=head_only)
        if head_only:
            logger.info(f'Epoch {epoch+1}: training classifier head only')
        train_loss, train_acc = train_one_epoch(train_loader, model, optimizer, criterion, epoch, scaler)

        should_eval = ((epoch + 1) % config.eval_every == 0) or (epoch + 1 == config.epochs)
        if should_eval:
            val_metrics = utils.evaluate_model(
                model, valid_loader, criterion, device,
                deepest=False,
                return_predictions=False,
                use_amp=use_amp,
                channels_last=config.channels_last and torch.cuda.is_available(),
            )
            val_loss = val_metrics['loss']
            val_acc = val_metrics['accuracy']
        else:
            val_loss = float('nan')
            val_acc = float('nan')
            logger.info(f'Epoch {epoch+1}: validation skipped (eval_every={config.eval_every})')
        scheduler.step()

        history['train_loss'].append(float(train_loss))
        history['train_acc'].append(float(train_acc))
        history['val_loss'].append(float(val_loss))
        history['val_acc'].append(float(val_acc))

        writer.add_scalar('train/loss', train_loss, epoch + 1)
        writer.add_scalar('train/acc', train_acc, epoch + 1)
        if should_eval:
            writer.add_scalar('val/loss', val_loss, epoch + 1)
            writer.add_scalar('val/acc', val_acc, epoch + 1)

        logger.info(
            f'Epoch {epoch+1}: train_loss={train_loss:.4f} train_acc={train_acc:.4%} '
            f'val_loss={val_loss:.4f} val_acc={val_acc:.4%}'
        )

        improved = should_eval and (val_acc > best_val_acc)
        if improved:
            best_val_acc = val_acc
        state = {
            'epoch': epoch + 1,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'best_val_acc': best_val_acc,
            'history': history,
            'config': vars(config),
            'classes': classes,
        }
        if scaler is not None:
            state['scaler_state'] = scaler.state_dict()
        utils.save_checkpoint(state, run_path, is_best=improved)

        if config.save_interval > 0 and (epoch + 1) % config.save_interval == 0:
            torch.save(state, run_path / f'checkpoint_epoch{epoch+1}.pth.tar')

        utils.save_json(run_path / 'history.json', history)
        utils.plot_training_curves(history, plots_dir / 'training_curves.png')

    logger.info(f'Best validation accuracy: {best_val_acc:.4%}')

    summary = {
        'best_val_acc': best_val_acc,
        'history': history,
        'optimizer': config.optimizer,
        'mixup_alpha': config.mixup_alpha,
        'amp': use_amp,
        'channels_last': bool(config.channels_last and torch.cuda.is_available()),
        'eval_every': config.eval_every,
    }

    if config.test_after_train:
        test_data, _ = get_split_dataset(config.data, config.data_path, 'test', config.img_size)
        test_loader = build_loader(test_data, shuffle=False)
        best_ckpt_path = run_path / 'best.pth.tar'
        utils.load_checkpoint(best_ckpt_path, model, device=device)
        test_metrics = utils.evaluate_model(
            model, test_loader, criterion, device,
            return_predictions=True,
            use_amp=use_amp,
            channels_last=config.channels_last and torch.cuda.is_available(),
        )
        summary['test'] = {k: v for k, v in test_metrics.items() if k not in ['targets', 'predictions', 'confusion_matrix']}
        utils.save_json(run_path / 'test_metrics.json', summary['test'])
        utils.plot_final_metrics(summary['test'], plots_dir / 'test_metrics.png')
        utils.plot_confusion_matrix(
            torch.tensor(test_metrics['confusion_matrix']).numpy(),
            plots_dir / 'test_confusion_matrix.png',
            title='Test confusion matrix',
        )
        logger.info(
            'Test metrics: '
            f"acc={test_metrics['accuracy']:.4%} precision={test_metrics['precision_macro']:.4f} "
            f"recall={test_metrics['recall_macro']:.4f} f1={test_metrics['f1_macro']:.4f}"
        )

    utils.save_json(run_path / 'summary.json', summary)
    writer.close()


if __name__ == '__main__':
    main()
