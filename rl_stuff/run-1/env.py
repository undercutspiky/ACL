import cPickle
import numpy as np
import scipy.stats
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


class Residual(nn.Module):
    def __init__(self, fan_in, fan_out, stride=1, w_init='kaiming_normal'):
        super(Residual, self).__init__()
        self.fan_in, self.fan_out = fan_in, fan_out
        self.conv1 = nn.Conv2d(fan_in, fan_out, 3, stride=stride, padding=1)
        self.conv2 = nn.Conv2d(fan_out, fan_out, 3, padding=1)
        self.expand_x = nn.Conv2d(fan_in, fan_out, 1)
        self.batch_norm1 = nn.BatchNorm2d(fan_in)
        self.batch_norm2 = nn.BatchNorm2d(fan_out)
        # Get weight initialization function
        w_initialization = getattr(nn.init, w_init)
        w_initialization(self.conv1.weight)
        nn.init.uniform(self.conv1.bias)
        w_initialization(self.conv2.weight)
        nn.init.uniform(self.conv2.bias)
        w_initialization(self.expand_x.weight)
        nn.init.uniform(self.expand_x.bias)

    def forward(self, x, downsample=False, train_mode=True):
        self.batch_norm1.training = train_mode
        self.batch_norm2.training = train_mode
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
    def __init__(self, width=1):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.res11 = Residual(16, 16*width)
        self.res12 = Residual(16*width, 16*width)
        self.res13 = Residual(16*width, 16*width)
        self.res14 = Residual(16*width, 16*width)
        self.res21 = Residual(16*width, 32*width, stride=2)
        self.res22 = Residual(32*width, 32*width)
        self.res23 = Residual(32*width, 32*width)
        self.res24 = Residual(32*width, 32*width)
        self.res31 = Residual(32*width, 64*width, stride=2)
        self.res32 = Residual(64*width, 64*width)
        self.res33 = Residual(64*width, 64*width)
        self.res34 = Residual(64*width, 64*width)
        self.final = nn.Linear(64*width, 10)

    def forward(self, x, train_mode=True):
        net = self.conv1(x)
        net = self.res11(net, train_mode=train_mode)
        net = self.res12(net, train_mode=train_mode)
        net = self.res13(net, train_mode=train_mode)
        net = self.res14(net, train_mode=train_mode)
        net = self.res21(net, train_mode=train_mode, downsample=True)
        net = self.res22(net, train_mode=train_mode)
        net = self.res23(net, train_mode=train_mode)
        net = self.res24(net, train_mode=train_mode)
        net = self.res31(net, train_mode=train_mode, downsample=True)
        net = self.res32(net, train_mode=train_mode)
        net = self.res33(net, train_mode=train_mode)
        net = self.res34(net, train_mode=train_mode)
        net = F.avg_pool2d(net, 8, 1)
        net = torch.squeeze(net)
        net = self.final(net)
        return net


class Env:
    def __init__(self):
        self.network = Net()
        self.network = self.network.cuda()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.network.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4, nesterov=True)
        self.batch_size = 128
        self.steps = 0
        self.train_x = []
        self.train_y = []
        for i in xrange(1, 5):
            dict_ = unpickle('../../cifar-10/cifar-10-batches-py/data_batch_' + str(i))
            if i == 1:
                self.train_x = np.array(dict_['data']) / 255.0
                self.train_y = dict_['labels']
            else:
                self.train_x = np.concatenate((self.train_x, np.array(dict_['data']) / 255.0), axis=0)
                self.train_y.extend(dict_['labels'])

        self.train_y = np.array(self.train_y)
        dict_ = unpickle('../../cifar-10/cifar-10-batches-py/data_batch_5')
        self.valid_x = np.array(dict_['data']) / 255.0
        self.valid_y = np.array(dict_['labels'])
        del dict_
        self.train_x = np.dstack((self.train_x[:, :1024], self.train_x[:, 1024:2048], self.train_x[:, 2048:]))
        self.train_x = np.reshape(self.train_x, [-1, 32, 32, 3])
        self.train_x = np.transpose(self.train_x, (0, 3, 1, 2))
        self.valid_x = np.dstack((self.valid_x[:, :1024], self.valid_x[:, 1024:2048], self.valid_x[:, 2048:]))
        self.valid_x = np.reshape(self.valid_x, [-1, 32, 32, 3])
        self.valid_x = np.transpose(self.valid_x, (0, 3, 1, 2))
        self.train_x = torch.from_numpy(self.train_x).float()
        self.valid_x = torch.from_numpy(self.valid_x).float().cuda()
        self.train_y = torch.from_numpy(self.train_y)
        self.valid_y = torch.from_numpy(self.valid_y).cuda()
        self.sequence = torch.randperm(self.train_x.size(0))
        self.train_x = self.train_x[self.sequence].cuda()
        self.train_y = self.train_y[self.sequence].cuda()

        print "Environment Initialized"

    def restore_state(self, state_name):
        self.network.load_state_dict(torch.load('./'+state_name+'.pth'))

    def save_state(self, state_name):
        torch.save(self.network.state_dict(), './'+state_name+'.pth')

    def get_stats(self, mat):
        a = []
        mat = mat.cpu().numpy()
        a.append(np.mean(mat))
        a.append(np.var(mat))
        a.append(np.mean(scipy.stats.skew(mat)))
        a.append(np.var(scipy.stats.skew(mat)))
        a.append(np.median(mat))
        return a

    def extract_state(self):
        state = []
        for i in xrange(1,4):
            for j in xrange(1,5):
                for k in xrange(1,3):
                    state.extend(self.get_stats(self.network.state_dict()['res'+str(i)+str(j)+'.conv'+str(k)+'.weight']))
        return state

    def take_action(self, batches):
        '''
        Trains the network on 1) batches received and 2) random batches
        :param batches: List of batches received from agent
        :return: The difference between rewards received from agent and adversary
        '''
        if self.steps > 50000:
            self.optimizer = optim.SGD(self.network.parameters(),
                                       lr=0.0001, momentum=0.9, weight_decay=5e-4, nesterov=True)
        elif self.steps > 20000:
            self.optimizer = optim.SGD(self.network.parameters(),
                                       lr=0.001, momentum=0.9, weight_decay=5e-4, nesterov=True)
        self.save_state('original')
        for batch in batches:
            self.optimizer.zero_grad()
            outputs = self.network(Variable(
                self.train_x[self.batch_size*batch: min(self.batch_size*batch +self.batch_size, self.train_x.size(0))]))
            loss = self.criterion(outputs, Variable(
                self.train_y[self.batch_size*batch: min(self.batch_size*batch +self.batch_size, self.train_x.size(0))]))
            loss.backward()
            self.optimizer.step()
        cursor, correct, total = (0,0,0)
        while cursor < len(self.valid_x):
            outputs = self.network(Variable(
                self.valid_x[cursor:min(cursor + self.batch_size, len(self.valid_x))]), train_mode=False)
            labels = self.valid_y[cursor:min(cursor + self.batch_size, len(self.valid_x))]
            _, predicted = torch.max(outputs.data, 1)
            total += len(labels)
            correct += (predicted == labels).sum()
            cursor += self.batch_size
        agent_reward = 100.0 * correct / total
        self.save_state('agent')
        self.restore_state('original')
        # Randomly select a sequence for adversary and train
        rand_seq = np.random.choice(40000, size=len(batches), replace=False)
        for batch in rand_seq:
            self.optimizer.zero_grad()
            outputs = self.network(Variable(
                self.train_x[self.batch_size*batch: min(self.batch_size*batch +self.batch_size, self.train_x.size(0))]))
            loss = self.criterion(outputs, Variable(
                self.train_y[self.batch_size*batch: min(self.batch_size*batch +self.batch_size, self.train_x.size(0))]))
            loss.backward()
            self.optimizer.step()
        cursor, correct, total = (0, 0, 0)
        while cursor < len(self.valid_x):
            outputs = self.network(Variable(
                self.valid_x[cursor:min(cursor + self.batch_size, len(self.valid_x))]), train_mode=False)
            labels = self.valid_y[cursor:min(cursor + self.batch_size, len(self.valid_x))]
            _, predicted = torch.max(outputs.data, 1)
            total += len(labels)
            correct += (predicted == labels).sum()
            cursor += self.batch_size
        adv_reward = 100.0 * correct / total
        if agent_reward > adv_reward:
            self.restore_state('agent')
            self.steps += len(batches)
        else:
            self.restore_state('original')

        return agent_reward, adv_reward, self.extract_state()

