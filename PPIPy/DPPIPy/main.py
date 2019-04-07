import subprocess
import torch
import torch.cuda as cutorch
import argparse

workDir = os.getcwd()
dataDir = workDir + '\\'

print('==> Options')

# global args
parser = argparse.ArgumentParser(description='Options:')
parser.add_argument('--device', default=0, help='set GPU device') # default cuda device index 0
parser.add_argument('-string', default='A',help='suffix to log files')
parser.add_argument('-saveModel', default=False, help='saves model if set to true')
parser.add_argument('-seed', default=1, help='set manual seed')

# data arg
parser.add_argument('-dataset', default='A', help='set training data')

# training args
parser.add_argument('-optimization', default='SGD', help='optimization method: SGD | ADAM')
parser.add_argument('-learningRate', default=0.01, help='initial learning rate')
parser.add_argument('-batchSize', default='100', help='mini batch training size')
parser.add_argument('-weightDecay', default=0, help='weight decay (SGD only)')
parser.add_argument('-momentum', default=0, help='momentum (SGD only)')
parser.add_argument('-epochs', default=100, help='number of epochs')
parser.add_argument('-epochID', default=1, help='starting epoch - used for resuming the run on servers')
parser.add_argument('-less_eval', default=False, help='evaluate every 10 epochs')
parser.add_argument('-crop', default=True, help='crop the sequence in true')
parser.add_argument('-cropLength', default=512, help='length of the cropped sequence')

# parse args
args = parser.parse_args()

# set the gpu
cutorch.set_device(args.device)

# create string used to save model and logs
saveString = str.format('%s-%s_crop%d-%s-rate%g', args.string, args.dataset, args.cropLength, args.optimization, args.learningRate)

# run args

# create/load log tensor to store evaluations
if args.epochID == 1:
    mylog = torch.Tensor(args.epochs+1,20).zero_()
    prlog = {}
else:
    mylog = torch.load(dataDir+'results/'+saveString+".pt")


epoch = args.epochID

for ep in range(epoch, args.epochs):
    a = 0
