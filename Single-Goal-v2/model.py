import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils as utils
import torch.nn.functional as F
import torch.optim as optim
from random import random

def conv2d_size_out(size, kernel_size=5, stride=2):
    return (size - (kernel_size - 1) - 1) // stride + 1

class DQN(nn.Module):
    def __init__(self, channel_in, height, width, action_dim):
        super(DQN, self).__init__()

        self.conv1 = nn.Conv2d(channel_in, 32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.convw = conv2d_size_out(conv2d_size_out(
            conv2d_size_out(width, 3, 1), 3, 2), 3, 1)
        self.convh = conv2d_size_out(conv2d_size_out(
            conv2d_size_out(height, 3, 1), 3, 2), 3, 1)

        self.l1 = nn.Linear(64 * self.convw * self.convh, 128)
        self.l2 = nn.Linear(128, 128)
        self.l_action = nn.Linear(128, action_dim)

    def forward(self, state):
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = F.relu(self.l1(x.view(x.size(0), -1)))
        x = F.relu(self.l2(x))
        action_values = self.l_action(x)
        return action_values


class Model(object):
    def __init__(self, lr, channel_in, height, width, action_dim, writer=None, device=None):
        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.writer = writer

        self.dqn1 = DQN(channel_in, height, width, action_dim).to(self.device)
        self.dqn2 = DQN(channel_in, height, width, action_dim).to(self.device)

        self.dqn1_optimizer = optim.Adam(self.dqn1.parameters(), lr=lr)
        self.dqn2_optimizer = optim.Adam(self.dqn2.parameters(), lr=lr)

    def select_action(self, state, dqn_num, epsilon=0.05):
        val = random()
        if val < epsilon:
            return int(random() * 4)
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
                if dqn_num == 0:
                    action = self.dqn1(state).squeeze()
                else:
                    action = self.dqn2(state).squeeze()
                return torch.argmax(action).cpu().item()

    def update(self, bs, replay_buffer, dqn_num, gamma=0.95):
        if len(replay_buffer) < bs:
            return
        states, actions, rewards, next_states, dones = replay_buffer.sample(bs)
        states = torch.FloatTensor(states).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        actions = np.stack((np.arange(bs), actions))
        if dqn_num == 0:
            q_values = self.dqn1(states)[actions]
            next_q_values = torch.max(self.dqn2(next_states), 1)[0]
        else:
            q_values = self.dqn2(states)[actions]
            next_q_values = torch.max(self.dqn1(next_states), 1)[0]

        loss = F.mse_loss(
            q_values, rewards + gamma * next_q_values * (1 - dones))
        if dqn_num == 0:
            self.dqn1_optimizer.zero_grad()
            loss.backward()
            self.dqn1_optimizer.step()
        
        else:
            self.dqn2_optimizer.zero_grad()
            loss.backward()
            self.dqn2_optimizer.step()

        if self.writer is not None:
            self.writer.add_histogram(
                'q_values', q_values.detach())

        return loss

    def save(self, directory, name):
        torch.save(self.dqn.state_dict(), '%s/%s_dqn.pth' % (directory, name))
        torch.save(self.dqn_target.state_dict(),
                   '%s/%s_dqn_target.pth' % (directory, name))
        torch.save(self.words, '%s/%s_words.pth' % (directory, name))
        torch.save(self.word_counter, '%s/%s_word_counter.pth' %
                   (directory, name))

    def load(self, directory, name):
        self.dqn.load_state_dict(torch.load('%s/%s_dqn.pth' % (directory, name),
                                            map_location=lambda storage, loc: storage))
        self.dqn_target.load_state_dict(torch.load('%s/%s_dqn_target.pth' % (directory, name),
                                                   map_location=lambda storage, loc: storage))
        self.words = torch.load('%s/%s_words.pth' % (directory, name))
        self.word_counter = torch.load(
            '%s/%s_word_counter.pth' % (directory, name))
