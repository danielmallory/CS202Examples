import torch.nn as nn
import torch
import opt

print('Creating the model')

# since there's no reshape model in Pytorch, we need a custom Flatten
# see https://discuss.pytorch.org/t/flatten-layer-of-pytorch-build-by-sequential-container/5983/3


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size()[0], -1)


num_features = 20
cropLength = 100

# Our overall model
model = nn.Sequential()

# convolutions
my_net = nn.Sequential(
    # convolution layer 1
    nn.Conv2d(num_features, 64, (1, 5), 1, 1, 0, 2),
    nn.BatchNorm2d(64),
    nn.ReLU(True),
    nn.AvgPool2d((1, 4)),

    # convolution layer 2
    nn.Conv2d(64, 128, (1, 5), 1, 1, 0, 2),
    nn.BatchNorm2d(128),
    nn.ReLU(True),
    nn.AvgPool2d((1, 4)),

    # convolution layer 3
    nn.Conv2d(128, 256, (1, 5), 1, 1, 0, 2),
    nn.BatchNorm2d(256),
    nn.ReLU(True),
    nn.AvgPool2d((1, 4)),

    # final convolution layer
    nn.Conv2d(256, 512, (1, 5), 1, 1, 0, 2),
    nn.BatchNorm2d(512),
    nn.ReLU(True),

    # reshape to output
    nn.AvgPool2d(1, opt.cropLength/64),
    Flatten()
)

# linear and FC layers
outH = 512
lin_net1 = nn.Sequential(
    nn.Linear(outH, outH),
    nn.BatchNorm1d(outH, affine=False),
    nn.ReLU(True)
)

lin_net2 = nn.Sequential(
    nn.Linear(outH, outH),
    nn.BatchNorm1d(outH, affine=False),
    nn.ReLU(True)
)

# copy our full-connected layers


# random layers
rand_layer1 = nn.Sequential(
)