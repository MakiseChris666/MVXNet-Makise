from optparse import OptionParser

parser = OptionParser(usage = 'python train.py <dataroot> [-n/--numepochs] [-r/--resume]\n'
                      + '<dataroot> is the root directory of KITTI dataset. The explanation is in README.')

parser.add_option('-n', '--numepochs', type = int, dest = 'numepochs',
                  help = 'Target number of training iterations. Default 10.', default = 10)
parser.add_option('-r', '--resume', type = int, dest = 'lastiter',
                  help = 'The number of last iteration. If specified, the training process will resume from the '
                         + 'checkpoint, which is checkpoints/epoch_{lastiter}.pkl.', default = 0)

options, args = parser.parse_args()