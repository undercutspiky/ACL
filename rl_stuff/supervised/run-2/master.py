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
        self.h = nn.LSTMCell(313, 128)
        self.action_head = nn.Linear(128, n_classes)
        # Initialize forget gate bias to 1
        self.h.bias_ih.data[self.h.bias_ih.size(0) / 4:self.h.bias_ih.size(0) / 2].fill_(1.0)
        self.h.bias_hh.data[self.h.bias_hh.size(0) / 4:self.h.bias_hh.size(0) / 2].fill_(1.0)

        self.hx = torch.zeros(1, 128)
        self.cx = torch.zeros(1, 128)

    def reset(self):
        self.hx = torch.zeros(1, 128)
        self.cx = torch.zeros(1, 128)

    def forward(self, x, length, train_mode=True):
        action_scores = []
        hx, cx = self.hx, self.cx
        hx, cx = Variable(hx.cuda()), Variable(cx.cuda())
        hx, cx = self.h(x, (hx, cx))
        actions = self.action_head(hx)
        action_scores.append(F.softmax(actions))
        for i in xrange(length - 1):
            hx, cx = self.h(Variable(torch.zeros(1, 313).cuda()), (hx, cx))
            actions = self.action_head(hx)
            action_scores.append(F.softmax(actions))
        self.hx, self.cx = hx.data, cx.data
        return action_scores


def select_action(state, out_length):
    probs = network(Variable(state.unsqueeze(0)), out_length)
    actions = []
    for i in xrange(len(probs)):
        action = probs[i].multinomial()
        actions.append(action.data)
    return actions, probs


def finish_episode(probs, targets):
    optimizer.zero_grad()
    loss = 0
    for i in xrange(len(probs)):
        loss += criterion(probs[i], Variable(torch.LongTensor([targets[i]]).cuda()))
    loss.backward()
    optimizer.step()


def save_state(state_name):
    torch.save(network.state_dict(), './' + state_name + '.pth')

network = Net()
network = network.cuda()
optimizer = optim.Adam(network.parameters(), lr=3e-2, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()
sequence = None
step = 0
for run in xrange(5):
    if sequence is not None:
        env = Env(sequence)
        network.reset()
    else:
        env = Env()
        sequence = env.sequence
    global_steps, epochs = 0, 0
    while epochs < 65:
        if epochs == 30:
            network.reset()
        state = torch.from_numpy(env.get_losses())
        state = state.cuda()
        # ad_reward, agent_reward = (0, -1)
        out_length = min(10 + step/10, 110)
        step += 1
        # count, batches = 0, []
        # while ad_reward > agent_reward:
        #     batches = select_action(state, out_length)
        #     ad_reward, agent_reward = env.take_action(batches)
        #     finish_episode(agent_reward - ad_reward)
        #     print ad_reward, agent_reward, [bat.cpu().numpy()[0][0] for bat in batches]
        #     count += 1
        batches, probs = select_action(state, out_length)
        targets, ad_reward, agent_reward = env.take_action(batches)
        finish_episode(probs, targets)
        global_steps += out_length
        epochs = global_steps//313
        print ('Accuracies - agent:%f best sequence:%f' % (agent_reward, ad_reward))
        print [bat.cpu().numpy()[0][0] for bat in batches]
        print ('%d global steps or ~ %d epochs done in run %d' % (global_steps, global_steps//313, run))

save_state('lstm_network')
