""" Config """
import argparse
import os
import shutil


class BaseConfig(argparse.Namespace):
    def check_exists(self, path):
        if os.path.exists(path) and not self.name.startswith("test"):
            while True:
                r = input("The path `{}` is already exists. Delete? [y/N/q]: ".format(path))
                r = r.lower().strip()
                if r == 'y':
                    shutil.rmtree(path)
                    print("\n!!! Remove original directory !!!\n")
                    break
                elif r in ['n', '']: # just enter is regarded as 'N'.
                    break
                elif r == 'q':
                    exit()

    def print_params(self, prtf=print):
        prtf("")
        prtf("Parameters:")
        for attr, value in sorted(vars(self).items()):
            prtf("{}={}".format(attr.upper(), value))
        prtf("")

    def as_markdown(self):
        """ Return configs as markdown format """
        text = "|name|value|  \n|-|-|  \n"
        for attr, value in sorted(vars(self).items()):
            text += "|{}|{}|  \n".format(attr, value)

        return text


class Config(BaseConfig):
    def build_parser(self):
        parser = argparse.ArgumentParser("Config")
        parser.add_argument('--name', required=True)
        parser.add_argument('--data', help='CIFAR10 (default) / CIFAR100', default='CIFAR10')
        parser.add_argument('--batch_size', type=int, default=100, help='default: 100')
        parser.add_argument('--lr', type=float, default=0.02, help='learning rate (default: 0.02)')
        parser.add_argument('--momentum', type=float, default=0.9)
        parser.add_argument('--print_freq', type=int, default=200, help='print frequency')
        parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
        parser.add_argument('--epochs', type=int, default=400,
                            help='# of training epochs (default: 400)')
        parser.add_argument('--init_channels', type=int, default=64,
                            help="doubling each block except the last (default: 64)")
        parser.add_argument('--gdrop_ratio', type=float, default=0.5,
                            help="ratio of global drop path (default: 0.5)")
        parser.add_argument('--p_ldrop', type=float, default=0.15,
                            help="local drop path probability (default: 0.15)")
        parser.add_argument('--dropout_probs', default="0.0, 0.1, 0.2, 0.3, 0.4",
                            help='dropout probs for each block with comma separated '
                                 '(default: 0.0, 0.1, 0.2, 0.3, 0.4)')
        parser.add_argument('--blocks', type=int, default=5, help='default: 5')
        parser.add_argument('--columns', type=int, default=3, help='default: 3')
        parser.add_argument('--seed', type=int, default=2, help='random seed')
        parser.add_argument('--workers', type=int, default=4, help='# of workers')
        parser.add_argument('--aug_lv', type=int, default=0,
                            help='data augmentation level (0~2). 0: no augmentation, '
                                 '1: horizontal mirroring + [-4, 4] translation, '
                                 '2: 1 + cutout.')
        parser.add_argument('--data_path', type=str, default='./data',
                    help='path to dataset (default: ./data)')
        parser.add_argument('--img_size', type=int, default=128,
                    help='input image size (for non-CIFAR datasets, default: 128)')
        # Оптимизатор
        parser.add_argument('--optimizer', type=str, default='sgd',
                    choices=['sgd', 'adam', 'adasmoothdelta'],
                    help='optimizer type (default: sgd)')

        # Возобновление обучения
        parser.add_argument('--resume', type=str, default=None,
                    help='path to checkpoint to resume training')

        # Интервал сохранения чекпоинтов (в эпохах)
        parser.add_argument('--save_interval', type=int, default=0,
                    help='save checkpoint every N epochs (0 = disable)')

        # Запускать ли тестирование на test после обучения
        parser.add_argument('--test_after_train', action='store_true', default=False,
                    help='run test on test set after training')

        # Режим только тестирования (без обучения)
        parser.add_argument('--test_only', action='store_true', default=False,
                    help='run only testing (requires --resume)')

        # EXPERIMENTS
        exp_parser = parser.add_argument_group('Experiment')
        exp_parser.add_argument('--off-drops', action='store_true', default=False,
                                help='turn off all dropout and droppath')
        exp_parser.add_argument('--gap', type=int, default=0, help='0: max-pool (default), '
                                '1: GAP - FC, 2: 1x1conv - GAP')
        exp_parser.add_argument('--init', default='xavier',
                                help='xavier (default) / he / torch (pytorch default)')
        exp_parser.add_argument('--pad', default='zero', help='zero (default) / reflect')
        exp_parser.add_argument('--doubling', default=False, action='store_true',
                                help='turn on 1x1 conv channel doubling')
        exp_parser.add_argument('--gdrop_type', default='ps-consist',
                                help='ps (per-sample, various gdrop per block) / '
                                'ps-consist (default; per-sample, consist global drop)')
        exp_parser.add_argument('--dropout_pos', default='CDBR',
                                help='CDBR (default; conv-dropout-BN-relu) / '
                                'CBRD (conv-BN-relu-dropout) / FD (fractal_block-dropout)')

        return parser

    def __init__(self):
        parser = self.build_parser()
        args = parser.parse_args()
        super().__init__(**vars(args))

        self.path = os.path.join("./runs", self.name)
        self.check_exists(self.path)

        self.data = self.data.lower().strip()
        self.data_path = args.data_path
        self.img_size = args.img_size
        self.optimizer = args.optimizer
        self.resume = args.resume
        self.save_interval = args.save_interval
        self.test_after_train = args.test_after_train
        self.test_only = args.test_only
        self.dropout_probs = [float(p) for p in self.dropout_probs.split(',')]
        self.consist_gdrop = self.gdrop_type == 'ps-consist'
        assert self.gdrop_type in ['ps', 'ps-consist']
        assert len(self.dropout_probs) == self.blocks

        # learning rate decay 4 times.
        # In the case of default epochs 400, lr milestone is = [200, 300, 350, 375].
        left = self.epochs // 2
        self.lr_milestone = [left]
        for i in range(3):
            left //= 2
            self.lr_milestone.append(self.lr_milestone[-1] + left)

        if self.off_drops:
            print("\n!!! Dropout and droppath are off !!!\n")
            for i in range(self.blocks):
                self.dropout_probs[i] = 0.
            self.p_ldrop = 0.
            self.gdrop_ratio = 0.


class TestConfig(BaseConfig):
    def build_parser(self):
        parser = argparse.ArgumentParser("Config")
        parser.add_argument('--name', required=True)
        parser.add_argument('--data', help='CIFAR10 (default) / CIFAR100', default='CIFAR10')
        parser.add_argument('--batch_size', type=int, default=200, help='default: 200')
        parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
        parser.add_argument('--init_channels', type=int, default=64,
                            help="doubling each block except the last (default: 64)")
        parser.add_argument('--blocks', type=int, default=5, help='default: 5')
        parser.add_argument('--columns', type=int, default=3, help='default: 3')
        parser.add_argument('--workers', type=int, default=4, help='# of workers')
        parser.add_argument('--test_set', type=str, default='valid', choices=['valid', 'test'])
        parser.add_argument('--model_file', type=str, default='best.pth.tar')

        # EXPERIMENTS
        exp_parser = parser.add_argument_group('Experiment')
        exp_parser.add_argument('--gap', type=int, default=0, help='0: max-pool (default), '
                                '1: GAP - FC, 2: 1x1conv - GAP')
        exp_parser.add_argument('--pad', default='zero', help='zero (default) / reflect')
        exp_parser.add_argument('--doubling', default=False, action='store_true',
                                help='turn on 1x1 conv channel doubling')

        return parser

    def __init__(self):
        parser = self.build_parser()
        args = parser.parse_args()
        super().__init__(**vars(args))

        self.data = self.data.lower().strip()
        self.data_path = './data/'
