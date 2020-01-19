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
    def __init__(self, height, width, channel_in, action_dim, input_size, device):
        super(DQN, self).__init__()

        self.device = device

        # Vision processing
        self.conv1 = nn.Conv2d(channel_in, 32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.convw = conv2d_size_out(conv2d_size_out(
            conv2d_size_out(width, 3, 1), 3, 2), 3, 1)
        self.convh = conv2d_size_out(conv2d_size_out(
            conv2d_size_out(height, 3, 1), 3, 2), 3, 1)

        # Language processing
        self.lstm_hidden_size = 128
        self.embedding = nn.Embedding(input_size, 32)
        self.lstm = nn.LSTM(32, self.lstm_hidden_size)

        # Gated-Attention layers
        self.attn_linear = nn.Linear(self.lstm_hidden_size, 64)

        self.l1 = nn.Linear(self.convw * self.convh * 64, 256)

        self.lstm_history = nn.LSTMCell(256, 256)

        self.l_action = nn.Linear(256, action_dim)

        self.to(self.device)

    def forward(self, inputs):
        state, advice, (hx, cx) = inputs

        # Image processing
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        image_rep = F.relu(self.conv3(x))

        # Language processing
        word_embedding = []
        for i in range(len(advice)):
            word_embedding.append(self.embedding(advice[i]))
        word_embedding = utils.rnn.pad_sequence(word_embedding)

        encoder_hidden = (torch.zeros(1, len(advice), self.lstm_hidden_size, requires_grad=True).to(self.device),
                          torch.zeros(1, len(advice), self.lstm_hidden_size, requires_grad=True).to(self.device))

        _, encoder_hidden = self.lstm(word_embedding, encoder_hidden)
        advice_rep = encoder_hidden[0].view(len(advice), -1)

        # Attention
        advice_attention = torch.sigmoid(self.attn_linear(advice_rep))

        # Gated-Attention
        advice_attention = advice_attention.unsqueeze(2).unsqueeze(3)
        advice_attention.expand(advice_attention.size(0),
                                64, self.convh, self.convw)

        x = image_rep * advice_attention
        x = x.view(x.size(0), -1)

        x = F.relu(self.l1(x))

        hx, cx = self.lstm_history(x, (hx, cx))

        action_values = self.l_action(hx)

        return action_values, (hx, cx)


class Model(object):
    def __init__(self, lr, height, width, channel_in, action_dim, input_size=128, writer=None, device=None):
        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.writer = writer

        self.input_size = input_size

        self.dqn1 = DQN(height, width, channel_in, action_dim,
                        self.input_size, self.device)
        self.dqn2 = DQN(height, width, channel_in,
                        action_dim, self.input_size, self.device)

        self.dqn1_optimizer = optim.Adam(self.dqn1.parameters(), lr=lr)
        self.dqn2_optimizer = optim.Adam(self.dqn2.parameters(), lr=lr)

        self.words = {}
        self.word_counter = 0

    def add_word(self, word):
        if word not in self.words:
            self.word_counter += 1
            self.words[word] = self.word_counter

    def advice_to_idx(self, advice):
        advice_idxes = []
        for i in range(len(advice)):
            advice_idx = torch.zeros(
                len(advice[i]), dtype=torch.long).to(self.device)
            for j in range(len(advice[i])):
                self.add_word(advice[i][j])
                advice_idx[j] = self.words[advice[i][j]]
            advice_idxes.append(advice_idx)
        return advice_idxes

    def select_action(self, state, advice, hx, cx, dqn_num, epsilon=0.05):
        val = random()
        if val < epsilon:
            return int(random() * 4), (hx, cx)
        else:
            with torch.no_grad():
                hx, cx = (torch.FloatTensor(hx).unsqueeze(0).to(
                    self.device), torch.FloatTensor(cx).unsqueeze(0).to(self.device))
                state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                advice = self.advice_to_idx([advice])
                if dqn_num == 0:
                    action, (hx, cx) = self.dqn1((state, advice, (hx, cx)))
                else:
                    action, (hx, cx) = self.dqn2((state, advice, (hx, cx)))
                return np.argmax(action.cpu().data.numpy().flatten()), \
                    (hx.cpu().detach().numpy(),
                     cx.cpu().detach().numpy())

    def update(self, bs, replay_buffer, dqn_num, gamma=0.9):
        if len(replay_buffer) < bs:
            return
        (states, advices), actions, rewards, next_states, dones, (hx,
                                                                  cx) = replay_buffer.sample(bs)
        states = torch.FloatTensor(states).to(self.device)
        advices = self.advice_to_idx(advices)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        actions = np.stack((np.arange(bs), actions))
        hx = torch.FloatTensor(hx).to(self.device)
        cx = torch.FloatTensor(cx).to(self.device)
        if dqn_num == 0:
            q_values, (hx, cx) = self.dqn1((states, advices, (hx, cx)))
            q_values = q_values[actions]
            next_q_values, _ = self.dqn2((next_states, advices, (hx, cx)))
            next_q_values = torch.max(next_q_values, 1)[0]
        else:
            q_values, _ = self.dqn2((states, advices, (hx, cx)))
            q_values = q_values[actions]
            next_q_values, _ = self.dqn1((next_states, advices, (hx, cx)))
            next_q_values = torch.max(next_q_values, 1)[0]

        expected_q_values = rewards + \
            (gamma * next_q_values * (1 - dones)).detach()

        loss = F.smooth_l1_loss(q_values, expected_q_values)

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
