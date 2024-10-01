import argparse

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()

parser.add_argument('--command', default='train', choices=['train'])
parser.add_argument('--dataset', default='mnist', choices=['mnist','cifar10'])
parser.add_argument('--num-classes', type=int, default=10)
parser.add_argument('--data-path', default='./data')
parser.add_argument('--label-corrupt-prob', type=float, default=0.0)

parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--batch-size', type=int, default=60, help="Batch size for training. Decreasing this value will make SGD favor flatter minima.")
parser.add_argument('--batch-size-for-validation', type=int, default=256, help="Batch size for validation and bound estimation. Enlarging this value will accelerate bound estimation at the risk of exceeding GPU memory.")
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--bound-epoch', type=int, default=29)
parser.add_argument('--learning-rate', type=float, default=0.01)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight-decay', type=float, default=0)
parser.add_argument('--width', type=float, default=None)
parser.add_argument('--test-freq', type=int, default=1)

parser.add_argument('--ad_lr', type=bool, default=False,
                    help='Whether to adjust learning_rate')
parser.add_argument('--bound', type=str2bool, nargs='?',
                        const=True, default=True, help='Whether to use perturbed training')
parser.add_argument('--fixinit', type=str2bool, nargs='?',
                        const=False, default=False, help='Whether to use perturbed training')
parser.add_argument('--early-stop', type=str2bool, nargs='?',
                        const=False, default=False, help='Whether to use perturbed training') 
parser.add_argument('--proxy', type=str2bool, nargs='?',
                        const=False, default=False, help='Whether to use perturbed training')                         
                        

parser.add_argument('--arch', default='fc1', choices=['fc1', 'lenet', 'alexnet', 'resnet','vgg', 'vit'])

parser.add_argument("--print-freq", default=1, type=int)
parser.add_argument("--valid-freq", default=1, type=int)
parser.add_argument("--resume", type=str, default=None)
parser.add_argument("--traj-reweight", type=float, nargs='+', default=[], help="Lambda(s) in Theorem 2 of the paper. The larger this value is, the more efforts will be put on decreasing the trajectory term. Enlarging this value too much will increase the penalty term.")

parser.add_argument('--name', default='', help='Experiment name.')

parser.add_argument("--k", "-k", "--parapllel-models", type=int, default=6, help="Number of parallel models for variance estimation")
parser.add_argument("--loss-upperbound", type=float, default=12.0, help="The scaling factor in the upperbound of Clipped Cross Entropy Loss in evaluation and bound estimation. Setting this value to u will set the upperbound to u * log C, where C is the number of classes.")
parser.add_argument("--train-loss-upperbound", type=float, default=None, help="The scaling factor in the upperbound of Clipped Cross Entropy Loss in training. Setting this value to u will set the upperbound to u * log C, where C is the number of classes.")
parser.add_argument("--bound-freq", type=int, default=None, help="The frequency of bound estimation. Defaults to `None`, which means only estimate the bound at the end of the whole training.")
parser.add_argument("--data-usage-for-bounds", type=float, default=1.0, help="The portion of data used when estimating the bounds. This can reduce the the time of bound estimation.")
parser.add_argument("--tolerance", type=float, default=1e-2, help="(Relative) tolerance when inverting Hessians.")
parser.add_argument("--validation-usage", type=float, default=0.33, help="The portion of validation data, split from the testing set, that is used for optimizing the bound.")
parser.add_argument("--dont-repeat", action='store_true', help="Skip experiments if there have been records with the same hyperparameter and longer epochs (they may have different random seeds).")

parser.add_argument("--dropout", type=float, default=0.0)
parser.add_argument("--warmup-epochs", type=int, default=None)
parser.add_argument("--optimizer", type=str, default="SGD", choices=["SGD", "AdamW"])
parser.add_argument("--scheduler", type=str, default=None, choices=[None, "cosine"])
parser.add_argument("--gradient-clipping", action="store_true")
parser.add_argument("--amp", action="store_true", help="Automatic Mixed Precision.")


def format_experiment_name(args):
    name = args.name
    if name != '':
        name += '_'
    name += args.dataset + '_'
    if args.label_corrupt_prob > 0:
        name += 'corrupt%g_' % args.label_corrupt_prob

    name += args.arch
    name += '_lr{0}_bs{1}'.format(args.learning_rate, args.batch_size)
    width = args.width
    if args.arch == 'resnet' and (width == 1.0 or width == 64):
        width = 64 
    name += f'_width{width}'
    if args.weight_decay > 0:
        name += '_Wd{0}'.format(args.weight_decay)
    else:
        name += '_NoWd'

    # name += f'_seed{args.seed}'

    return name


def parse_args():
    import random
    args = parser.parse_args()
    if args.seed is None or args.seed < 0:
        args.seed  = random.randint(0, 100000)
    if args.bound_freq <= 0:
        args.bound_freq = None
    if args.arch == 'fc1':
        args.width = int(args.width) if args.width is not None else 512
    elif args.arch == 'resnet':
        args.width = args.width if args.width is not None else 64
    elif args.arch == 'vit':
        if args.width > 10:
            args.width = args.width / 64
        args.width = args.width if args.width is not None else 1.0
    else:
        raise NotImplemented(args.arch)
    args.amp = True
    args.exp_name = format_experiment_name(args)
    return args
