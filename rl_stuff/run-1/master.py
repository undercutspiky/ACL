import cPickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import autograd
from torch.autograd import Variable
from env import Env


class Net(nn.Module):
    def __init__(self, n_classes=313):
        super(Net, self).__init__()
        self.h = nn.LSTMCell(120, 100)
        self.action_head = nn.Linear(100, n_classes)
        # Initialize forget gate bias to 1
        self.h.bias_ih.data[self.h.bias_ih.size(0) / 4:self.h.bias_ih.size(0) / 2].fill_(1.0)
        self.h.bias_hh.data[self.h.bias_hh.size(0) / 4:self.h.bias_hh.size(0) / 2].fill_(1.0)

        self.saved_actions = []
        self.hx = torch.randn(1, 100)
        self.cx = torch.randn(1, 100)

    def forward(self, x, length, train_mode=True):
        action_scores = []
        hx, cx = self.hx, self.cx
        hx, cx = Variable(hx.cuda()), Variable(cx.cuda())
        for i in xrange(length):
            hx, cx = self.h(x, (hx, cx))
            actions = self.action_head(self.hx)
            action_scores.append(F.softmax(actions))
        self.hx, self.cx = hx, cx
        return action_scores


def select_action(state, out_length):
    probs = network(Variable(state.unsqueeze(0)), out_length)
    actions = []
    for i in xrange(len(probs)):
        action = probs[i].multinomial()
        network.saved_actions.append(action)
        actions.append(action.data)
    return actions


def finish_episode(reward):
    saved_actions = network.saved_actions
    for action in saved_actions:
        action.reinforce(reward)
    optimizer.zero_grad()
    autograd.backward(network.saved_actions, [None for _ in network.saved_actions])
    optimizer.step()
    del network.saved_actions[:]

network = Net()
network = network.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(network.parameters(), lr=3e-2, weight_decay=5e-4)
env = Env()

for step in xrange(1000):
    state = torch.from_numpy(env.extract_state())
    state = state.cuda()
    ad_reward, agent_reward = (0, -1)
    out_length = 10 + step/10
    count = 0
    while ad_reward > agent_reward:
        batches = select_action(state, out_length)
        ad_reward, agent_reward, state = env.take_action(batches)
        finish_episode(agent_reward - ad_reward)
        print ad_reward, agent_reward
        count += 1
    print ('Accuracies after %d tries - agent:%f adversary:%f' % (count, agent_reward, ad_reward))

