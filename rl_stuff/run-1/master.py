import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import autograd
from torch.autograd import Variable
from env import Env


class Net(nn.Module):
    def __init__(self, n_classes=313, n_hidden=256):
        super(Net, self).__init__()
        self.n_hidden = n_hidden
        self.h = nn.LSTMCell(313+120, self.n_hidden)
        self.action_head = nn.Linear(self.n_hidden, n_classes)
        # Initialize forget gate bias to 1
        self.h.bias_ih.data[self.h.bias_ih.size(0) / 4:self.h.bias_ih.size(0) / 2].fill_(1.0)
        self.h.bias_hh.data[self.h.bias_hh.size(0) / 4:self.h.bias_hh.size(0) / 2].fill_(1.0)

        self.saved_actions = []
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
        actions.append(action)
    network.saved_actions.append(actions)
    return [a.data for a in actions]


def finish_episode(rewards):
    rewards = torch.Tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)
    optimizer.zero_grad()
    saved_actions = network.saved_actions
    for i in xrange(len(saved_actions)):
        for a in saved_actions[i]:
            a.reinforce(rewards[i])
    a = network.saved_actions[0]
    for i in xrange(1, len(network.saved_actions)):
        a.extend(network.saved_actions[i])
    autograd.backward(a, [None for _ in a])
    nn.utils.clip_grad_norm(network.parameters(), 0.5)
    optimizer.step()
    del network.saved_actions[:]


def save_state(state_name):
    torch.save(network.state_dict(), './' + state_name + '.pth')

network = Net()
network = network.cuda()
optimizer = optim.SGD(network.parameters(), lr=0.001, momentum=0.7, weight_decay=5e-4, nesterov=True)
sequence = None
step = 0
for run in xrange(5):
    if sequence is not None:
        env = Env(sequence)
        network.reset()
    else:
        env = Env()
        sequence = env.sequence
    global_steps, epochs, rewards, count_steps, count_points = 0, 0, [], 0, 0
    while global_steps//313 < 75:
        state = env.get_losses()
        state.extend(env.extract_state())
        state = torch.from_numpy(np.array(state)).cuda()
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
        batches = select_action(state, out_length)
        ad_reward, agent_reward = env.take_action(batches)
        rewards.append(agent_reward - ad_reward)
        global_steps += out_length
        count_points += out_length
        count_steps += 1
        if count_points > 300 or count_steps > 10:
            finish_episode(rewards)
            count_points, count_steps, rewards = 0, 0, []
        print ('Accuracies - agent:%f adversary:%f' % (agent_reward, ad_reward))
        print [bat.cpu().numpy()[0][0] for bat in batches]
        print ('%d global steps or ~ %d epochs done in run %d' % (global_steps, global_steps//313, run))

save_state('lstm_network')
np.save('sequence', sequence)

