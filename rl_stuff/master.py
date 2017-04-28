import cPickle, gzip
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

def unpickle(file):
    fo = open(file, 'rb')
    dict_ = cPickle.load(fo)
    fo.close()
    return dict_

# Load CIFAR-10 data
train_x = []
train_y = []
for i in xrange(1, 5):
    dict_ = unpickle('../cifar-10/cifar-10-batches-py/data_batch_' + str(i))
    if i == 1:
        train_x = np.array(dict_['data'])/255.0
        train_y = dict_['labels']
    else:
        train_x = np.concatenate((train_x, np.array(dict_['data'])/255.0), axis=0)
        train_y.extend(dict_['labels'])

train_y = np.array(train_y)
dict_ = unpickle('../cifar-10/cifar-10-batches-py/data_batch_5')
valid_x = np.array(dict_['data'])/255.0
valid_y = np.eye(10)[dict_['labels']]
train_y = np.eye(10)[train_y]
del dict_
train_x = torch.from_numpy(train_x).cuda()
valid_x = torch.from_numpy(valid_x).cuda()
train_y = torch.from_numpy(train_y).cuda()
valid_y = torch.from_numpy(valid_y).cuda()


class Residual(nn.Module):
    def __init__(self, fan_in, fan_out, w_init='kaiming_normal'):
        super(Residual, self).__init__()
        self.fan_in, self.fan_out = fan_in, fan_out
        self.conv1 = nn.Conv2d(fan_in, fan_out, 3)
        self.conv2 = nn.Conv2d(fan_in, fan_out, 3)
        self.expand_x = nn.Conv2d(fan_in, fan_out, 1)
        self.batch_norm1 = nn.BatchNorm1d(fan_out)
        self.batch_norm2 = nn.BatchNorm1d(fan_out)
        # Get weight initialization function
        w_initialization = getattr(nn.init, w_init)
        w_initialization(self.conv1.weight)
        nn.init.uniform(self.conv1.bias)
        w_initialization(self.conv2.weight)
        nn.init.uniform(self.conv2.bias)
        w_initialization(self.expand_x.weight)
        nn.init.uniform(self.expand_x.bias)

    def forward(self, x, downsample=False, train_mode=True):
        self.batch_norm.training = train_mode
        h = self.conv1(F.leaky_relu(self.batch_norm1(x)))
        h = self.conv2(F.leaky_relu(self.batch_norm2(h)))
        if downsample:
            x_new = F.avg_pool2d(x, 2, 2)
            if self.fan_in != self.fan_out:
                return h + self.expand_x(x_new)
            return h + x_new
        if self.fan_in != self.fan_out:
            return h + self.expand_x(x)
        return h + x


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.res11 = Residual(16, 16)
        self.res12 = Residual(16, 16)
        self.res13 = Residual(16, 16)
        self.res21 = Residual(16, 32)
        self.res22 = Residual(32, 32)
        self.res23 = Residual(32, 32)
        self.res31 = Residual(32, 64)
        self.res32 = Residual(64, 64)
        self.final = nn.Linear(64, 10)

    def forward(self, x, train_mode=True, get_t=False):
        net = self.conv1(x)
        stage1 = self.res13(self.res12(self.res11(net)))
        stage2 = self.res21(self.res22(self.res23(stage1)))
        stage3 = self.res31(self.res32(stage2))
        net = F.avg_pool2d(stage3, 8, 1)
        net = self.final(net)
        return net


network = Net()
network = network.cuda()
print network
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(network.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0002)

epochs = 150
batch_size = 128

for epoch in xrange(1, epochs + 1):

    if epoch > 120:
        optimizer = optim.SGD(network.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0002)
    elif epoch > 60:
        optimizer = optim.SGD(network.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0002)
    cursor = 0
    while cursor < len(train_x):
        optimizer.zero_grad()
        outputs, t_cost = network(Variable(train_x[cursor:min(cursor + batch_size, len(train_x))]))
        loss = criterion(outputs, Variable(train_y[cursor:min(cursor + batch_size, len(train_x))]))
        loss.backward()
        optimizer.step()
        cursor += batch_size

    cursor = 0
    correct = 0
    total = 0
    while cursor < len(valid_x):
        outputs = network(Variable(valid_x[cursor:min(cursor + batch_size, len(valid_x))]), train_mode=False)
        labels = valid_y[cursor:min(cursor + batch_size, len(valid_x))]
        _, predicted = torch.max(outputs.data, 1)
        total += len(labels)
        correct += (predicted == labels).sum()
        cursor += batch_size

    print('For epoch %d \tAccuracy on valid set: %f %%' % (epoch, 100.0 * correct / total))
