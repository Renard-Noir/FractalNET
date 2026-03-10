""" Trainer """
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from config import Config
import utils
from fractal import FractalNet
from datasets import get_dataset
import argparse
import json
from pathlib import Path
from datetime import datetime
from collections import OrderedDict
import torch.optim as optim
# Трансформации для теста (аналогично валидационным)
from torchvision import transforms
import optimizers


config = Config()
device = torch.device("cuda")

# tensorboard
writer = SummaryWriter(log_dir=os.path.join(config.path, "tb"))
writer.add_text('config', config.as_markdown(), 0)

# logger
logger = utils.get_logger(os.path.join(config.path, "{}.log".format(config.name)))
logger.info("Run options = {}".format(sys.argv))
config.print_params(logger.info)

# copy scripts
utils.copy_scripts("*.py", config.path)


def main():
    if config.test_only:
        if not config.resume:
            raise ValueError("--test_only requires --resume")
        # загружаем модель из чекпоинта
        ckpt = load_checkpoint(Path(config.resume), model, device=device)
        # создаём test_loader (как выше)
        # оцениваем и выводим результат
        test_loss, test_acc, per_class_acc = evaluate_with_per_class(...)
        print(f"Test loss: {test_loss:.4f}, accuracy: {test_acc:.4%}")
        return
    logger.info("Logger is set - training start")

    # set gpu device id
    logger.info("Set GPU device {}".format(config.gpu))
    torch.cuda.set_device(config.gpu)

    # set seed
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    torch.backends.cudnn.benchmark = True

    # get dataset
    train_data, valid_data, data_shape = get_dataset(config.data, config.data_path, config.aug_lv)
    # Параметры нормализации (должны совпадать с используемыми в get_dataset)
    if config.data == 'stanford_dogs':
        MEAN = [0.485, 0.456, 0.406]
        STD = [0.229, 0.224, 0.225]
        img_size = config.img_size if hasattr(config, 'img_size') else 128
        test_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])
        test_root = Path(config.data_path) / 'test' / 'images'
        test_dataset = datasets.ImageFolder(test_root, transform=test_transform)
        test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=config.batch_size,
                                              shuffle=False,
                                              num_workers=config.workers,
                                              pin_memory=True)
    else:
        test_loader = None   # для CIFAR тестовая выборка уже есть в valid_data

    # build model
    criterion = nn.CrossEntropyLoss().to(device)
    model = FractalNet(data_shape, config.columns, config.init_channels,
                       p_ldrop=config.p_ldrop, dropout_probs=config.dropout_probs,
                       gdrop_ratio=config.gdrop_ratio, gap=config.gap,
                       init=config.init, pad_type=config.pad, doubling=config.doubling,
                       dropout_pos=config.dropout_pos, consist_gdrop=config.consist_gdrop)
    model = model.to(device)

    # model size
    m_params = utils.param_size(model)
    logger.info("Models:\n{}".format(model))
    logger.info("Model size (# of params) = {:.3f} M".format(m_params))

    # weights optimizer
    #optimizer = torch.optim.SGD(model.parameters(), config.lr, momentum=config.momentum)
    if config.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), config.lr, momentum=config.momentum)
    elif config.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    elif config.optimizer == 'adasmoothdelta':
        optimizer = AdaSmoothDelta(model.parameters(), lr=config.lr)
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer}")

    # setup data loader
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=config.batch_size,
                                               shuffle=True,
                                               num_workers=config.workers,
                                               pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(valid_data,
                                               batch_size=config.batch_size,
                                               shuffle=False,
                                               num_workers=config.workers,
                                               pin_memory=True)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, config.lr_milestone)

    best_top1 = 0.

    # Получаем список классов из train_dataset (для Stanford Dogs)
    if config.data == 'stanford_dogs':
        classes = train_dataset.classes
    else:
        classes = []   # для CIFAR не нужны

    # Генерация model_id (можно использовать имя эксперимента)
    model_id = config.name
    meta_path = Path(config.path) / "metadata.json"
    entry, container = load_or_create_metadata(meta_path, model_id, config, classes)
    entry["start_time"] = datetime.now().strftime('%Y%m%d_%H%M%S')
    persist_metadata(meta_path, container)

    start_epoch = 0
    best_val_acc = 0.0

    if config.resume:
        resume_path = Path(config.resume)
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume file not found: {resume_path}")
        ckpt = load_checkpoint(resume_path, model, optimizer, device)
        # предполагаем, что в чекпоинте есть поля 'epoch' и 'best_val_acc'
        start_epoch = ckpt.get('epoch', 0)
        best_val_acc = ckpt.get('best_val_acc', 0.0)
        # если есть история, можно загрузить её из метаданных (они уже должны быть)
        logger.info(f"Resumed from epoch {start_epoch}, best_val_acc={best_val_acc:.4f}")

    # training loop
    for epoch in range(config.epochs):
        lr_scheduler.step()

        # training
        train(train_loader, model, optimizer, criterion, epoch)

        # validation
        cur_step = (epoch+1) * len(train_loader)
        top1 = validate(valid_loader, model, criterion, epoch, cur_step)

        # сохраняем метаданные
        entry["history"]["train_loss"].append(train_loss)
        entry["history"]["train_acc"].append(train_acc)
        entry["history"]["val_loss"].append(val_loss)
        entry["history"]["val_acc"].append(val_acc)
        entry["timestamp"] = datetime.now().strftime('%Y%m%d_%H%M%S')
        persist_metadata(meta_path, container)

        # Сохранение чекпоинтов по интервалу
        if config.save_interval > 0 and (epoch + 1) % config.save_interval == 0:
            ckpt_path = os.path.join(config.path, f'checkpoint_epoch{epoch+1}.pth.tar')
            torch.save({
                'epoch': epoch + 1,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'config': config
            }, ckpt_path)
            entry["last_checkpoint_epoch"] = epoch + 1
            persist_metadata(meta_path, container)
            logger.info(f"Checkpoint saved: {ckpt_path}")

        # save
        if best_top1 < top1:
            best_top1 = top1
            is_best = True
            entry["best"] = {
                "epoch": epoch+1,
                "val_acc": best_val_acc,
                "model_path": os.path.join(config.path, 'best.pth.tar')
            }
            persist_metadata(meta_path, container)
        else:
            is_best = False
        utils.save_checkpoint(model.state_dict(), config.path, is_best)

        print("")

    logger.info("Final best Prec@1 = {:.4%}".format(best_top1))

    if config.test_after_train and test_loader is not None:
        logger.info("Evaluating best model on test set...")
        # загружаем лучшую модель
        best_path = os.path.join(config.path, 'best.pth.tar')
        if os.path.exists(best_path):
            model.load_state_dict(torch.load(best_path, map_location=device))
        else:
            # используем текущую модель (финальную)
            pass

        test_loss, test_acc, per_class_acc = evaluate_with_per_class(
            model, test_loader, criterion, device, num_classes
        )
        logger.info(f"Test: Loss {test_loss:.4f}, Acc {test_acc:.4%}")

        # Сохраняем результаты в metadata
        entry["final"] = {
            "test_loss": test_loss,
            "test_acc": test_acc,
            "per_class_acc": per_class_acc
        }
        persist_metadata(meta_path, container)


def train(train_loader, model, optimizer, criterion, epoch):
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    losses = utils.AverageMeter()

    cur_step = epoch*len(train_loader)
    cur_lr = optimizer.param_groups[0]['lr']
    logger.info("Epoch {} LR {}".format(epoch+1, cur_lr))
    writer.add_scalar('train/lr', cur_lr, cur_step)

    model.train()

    for step, (X, y) in enumerate(train_loader):
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        N = X.size(0)

        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        prec1, prec5 = utils.accuracy(logits, y, topk=(1, 5))
        losses.update(loss.item(), N)
        top1.update(prec1.item(), N)
        top5.update(prec5.item(), N)

        if step % config.print_freq == 0 or step == len(train_loader)-1:
            logger.info(
                "Train: [{:3d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                    epoch+1, config.epochs, step, len(train_loader)-1, losses=losses,
                    top1=top1, top5=top5))

        writer.add_scalar('train/loss', loss.item(), cur_step)
        writer.add_scalar('train/top1', prec1.item(), cur_step)
        writer.add_scalar('train/top5', prec5.item(), cur_step)
        cur_step += 1

    logger.info("Train: [{:3d}/{}] Final Prec@1 {:.4%}".format(epoch+1, config.epochs, top1.avg))


def validate(valid_loader, model, criterion, epoch, cur_step, deepest=False):
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    losses = utils.AverageMeter()

    model.eval()

    with torch.no_grad():
        for step, (X, y) in enumerate(valid_loader):
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            N = X.size(0)

            logits = model(X, deepest=deepest)
            loss = criterion(logits, y)

            prec1, prec5 = utils.accuracy(logits, y, topk=(1, 5))
            losses.update(loss.item(), N)
            top1.update(prec1.item(), N)
            top5.update(prec5.item(), N)

            if step % config.print_freq == 0 or step == len(valid_loader)-1:
                logger.info(
                    "Valid: [{:3d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                    "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                        epoch+1, config.epochs, step, len(valid_loader)-1, losses=losses,
                        top1=top1, top5=top5))

    writer.add_scalar('val/loss', losses.avg, cur_step)
    writer.add_scalar('val/top1', top1.avg, cur_step)
    writer.add_scalar('val/top5', top5.avg, cur_step)

    logger.info("Valid: [{:3d}/{}] Final Prec@1 {:.4%}".format(epoch+1, config.epochs, top1.avg))

    return top1.avg


if __name__ == "__main__":
    main()
