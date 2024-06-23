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
parser.add_argument('--data-path', default='./datasets')
parser.add_argument('--label-corrupt-prob', type=float, default=0)

parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--batch-size', type=int, default=60)
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--bound-epoch', type=int, default=29)
parser.add_argument('--learning-rate', type=float, default=0.01)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight-decay', type=float, default=0)
parser.add_argument('--width', type=int, default=512)
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
parser.add_argument('--early_stop', type=str2bool, nargs='?',
                        const=False, default=False, help='Whether to use perturbed training') 
parser.add_argument('--proxy', type=str2bool, nargs='?',
                        const=False, default=False, help='Whether to use perturbed training')                         
                        

parser.add_argument('--arch', default='fc1', choices=['fc1', 'lenet', 'alexnet', 'resnet','vgg'])

parser.add_argument("--print_freq", default=1, type=int)
parser.add_argument("--valid_freq", default=1, type=int)
parser.add_argument("--resume", action="store_true")

parser.add_argument('--name', default='', help='Experiment name')

parser.add_argument("--k", type=int, default=5, help="The number of parallel models for variance estimation")
parser.add_argument("--p", type=float, default=0.5, help="The relative size of each training set in the parallel trainings for variance estimation")
parser.add_argument("--trajectories-for-optimization", type=int, default=3)


def format_experiment_name(args):
    name = args.name
    if name != '':
        name += '_'
    name += args.dataset + '_'
    if args.label_corrupt_prob > 0:
        name += 'corrupt%g_' % args.label_corrupt_prob

    name += args.arch
    name += '_lr{0}_bs{1}'.format(args.learning_rate, args.batch_size)
    if args.weight_decay > 0:
        name += '_Wd{0}'.format(args.weight_decay)
    else:
        name += '_NoWd'

    return name


def parse_args():
    args = parser.parse_args()
    args.exp_name = format_experiment_name(args)
    return args