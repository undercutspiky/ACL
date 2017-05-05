import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import time
from torch.autograd import Variable
from env_test import Env


class Net(nn.Module):
    def __init__(self, n_classes=313, n_hidden=256):
        super(Net, self).__init__()
        self.n_hidden = n_hidden
        self.h = nn.LSTMCell(313+120, self.n_hidden)
        self.action_head = nn.Linear(self.n_hidden, n_classes)
        # Initialize forget gate bias to 1
        self.h.bias_ih.data[self.h.bias_ih.size(0) / 4:self.h.bias_ih.size(0) / 2].fill_(1.0)
        self.h.bias_hh.data[self.h.bias_hh.size(0) / 4:self.h.bias_hh.size(0) / 2].fill_(1.0)

        self.hx = torch.zeros(1, self.n_hidden)
        self.cx = torch.zeros(1, self.n_hidden)

    def reset(self):
        self.hx = torch.zeros(1, self.n_hidden)
        self.cx = torch.zeros(1, self.n_hidden)

    def forward(self, x, length, train_mode=True):
        action_scores = []
        hx, cx = self.hx, self.cx
        hx, cx = Variable(hx.cuda()), Variable(cx.cuda())
        hx, cx = self.h(x, (hx, cx))
        actions = self.action_head(hx)
        action_scores.append(F.softmax(actions))
        for i in xrange(length - 1):
            hx, cx = self.h(Variable(torch.zeros(1, 313+120).cuda()), (hx, cx))
            actions = self.action_head(hx)
            action_scores.append(actions)
        self.hx, self.cx = hx.data, cx.data
        return action_scores


def select_action(state, out_length):
    log_probs = network(Variable(state.unsqueeze(0)), out_length)
    actions = []
    for i in xrange(len(log_probs)):
        action = F.softmax(log_probs[i]).multinomial()
        actions.append(action.data)
    return actions


def restore_state(state_name):
    network.load_state_dict(torch.load('./' + state_name + '.pth'))

network = Net()
network = network.cuda()
restore_state('lstm_network')
sequence = np.load('sequence.npy')

for run in xrange(5):
    network.reset()
    env = Env(sequence)
    while not os.path.exists('./state' + str(run) + '.pth'):
        time.sleep(1)
    env.restore_state('./state' + str(run))
    global_steps = 0
    accuracies = []
    while global_steps//313 < 75:
        out_length = 105
        batches = np.random.choice(313, size=out_length, replace=False)
        accuracy = env.take_action(batches, numpy=True)
        accuracies.append(accuracy)
        global_steps += out_length
        print ('Random accuracy on validation set = %f' % accuracy)
        print [bat.cpu().numpy()[0][0] for bat in batches]

    np.save('rand-accuracies'+str(run), accuracies)
