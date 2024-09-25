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
parser.add_argument('--label-corrupt-prob', type=float, default=0)

parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--batch-size', type=int, default=60)
parser.add_argument('--batch-size-for-validation', type=int, default=256)
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--bound-epoch', type=int, default=29)
parser.add_argument('--learning-rate', type=float, default=0.01)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight-decay', type=float, default=0)
parser.add_argument('--width', type=int, default=None)
parser.add_argument('--clip', type=float, default=0)
parser.add_argument('--clip_factor', type=float, default=0.1)
parser.add_argument('--stra', type=int, default=1)
parser.add_argument('--clip_start', type=int, default=0)
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
                        

parser.add_argument('--arch', default='fc1', choices=['fc1', 'lenet', 'alexnet', 'resnet','vgg'])

parser.add_argument("--print-freq", default=1, type=int)
parser.add_argument("--valid-freq", default=1, type=int)
parser.add_argument("--resume", type=str, default=None)
parser.add_argument("--traj-reweight", type=float, nargs='+', default=[])

parser.add_argument('--name', default='', help='Experiment name')

parser.add_argument("--k", "-k", "--parapllel-models", type=int, default=6, help="The number of parallel models for variance estimation")
parser.add_argument("--loss-upperbound", type=float, default=12.0, help="The scaling factor in the upperbound of Clipped Cross Entropy Loss in evaluation and bound estimation. Setting this value to u will set the upperbound to u * log C, where C is the number of classes.")
parser.add_argument("--train-loss-upperbound", type=float, default=None, help="The scaling factor in the upperbound of Clipped Cross Entropy Loss in training. Setting this value to u will set the upperbound to u * log C, where C is the number of classes.")
parser.add_argument("--bound-freq", type=int, default=None, help="The frequency of bound estimation. Defaults to `None`, which means only estimate the bound at the end of the whole training.")
parser.add_argument("--data-usage-for-bounds", type=float, default=1.0, help="The portion of data used when estimating the bounds. This can reduce the the time of bound estimation.")
parser.add_argument("--tolerance", type=float, default=1e-2)
parser.add_argument("--validation-usage", type=float, default=0.33, help="The portion of validation data, split from the testing set, that is used for optimizing the bound.")
parser.add_argument("--dont-repeat", action='store_true')


def format_experiment_name(args):
    name = args.name
    if name != '':
        name += '_'
    name += args.dataset + '_'
    if args.label_corrupt_prob > 0:
        name += 'corrupt%g_' % args.label_corrupt_prob

    name += args.arch
    name += '_lr{0}_bs{1}'.format(args.learning_rate, args.batch_size)
    name += f'_width{args.width}'
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
    else:
        raise NotImplemented(args.arch)
    args.exp_name = format_experiment_name(args)
    return args
