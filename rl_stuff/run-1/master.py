import cPickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import autograd
from torch.autograd import Variable
from collections import namedtuple
from env import Env


class Net(nn.Module):
    def __init__(self, n_classes=313):
        super(Net, self).__init__()
        self.h = nn.LSTMCell(120, 100)
        self.value_head = nn.Linear(100, n_classes)
        self.action_head = nn.Linear(100, n_classes)
        # Initialize forget gate bias to 1
        self.h.bias_ih.data[self.h.bias_ih.size(0) / 4:self.h.bias_ih.size(0) / 2].fill_(1.0)
        self.h.bias_hh.data[self.h.bias_hh.size(0) / 4:self.h.bias_hh.size(0) / 2].fill_(1.0)

        self.saved_actions = []

    def forward(self, x, length, train_mode=True):
        action_scores = []
        state_values = []
        for i in xrange(length):
            hx, cx = self.h(x, (hx, cx))
            values = self.value_head(hx)
            actions = self.action_head(hx)
            action_scores.append(F.softmax(actions))
            state_values.append(values)
        return action_scores, state_values


def select_action(state, out_length):
    probs, state_value = network(Variable(state.unsqueeze(0)), out_length)
    actions = []
    for i in xrange(len(probs)):
        action = probs[i].multinomial()
        network.saved_actions.append(SavedAction(action, state_value[i]))
        actions.append(action.data)
    return actions


def finish_episode(reward):
    saved_actions = network.saved_actions
    value_loss = 0
    for (action, value), r in zip(saved_actions):
        action.reinforce(reward)
        value_loss += F.smooth_l1_loss(value, Variable(torch.Tensor([reward])))
    optimizer.zero_grad()
    final_nodes = [value_loss] + list(map(lambda p: p.action, saved_actions))
    gradients = [torch.ones(1)] + [None] * len(saved_actions)
    autograd.backward(final_nodes, gradients)
    optimizer.step()
    del network.saved_actions[:]

network = Net()
network = network.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(network.parameters(), lr=3e-2, weight_decay=5e-4)
SavedAction = namedtuple('SavedAction', ['action', 'value'])
env = Env()

for step in xrange(1000):
    state = env.extract_state()
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

