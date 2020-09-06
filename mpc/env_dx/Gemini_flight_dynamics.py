import torch
import numpy as np
# import bagread
# import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import time


NUM_HISTORY = 2
NUM_INPUTS = (NUM_HISTORY + 1) * 13
NUM_OUTPUTS = 18
NUM_HIDDEN_UNITS = 200
NUM_HIDDEN_LAYERS = 2
NUM_ENSEMBLE = 1

PATH = ['flight_model_net1_ctrl_256_1000_2layers_2his_noval.pth',
        'flight_model_net2_ctrl_256_1000_2layers_2his_noval.pth',
        'flight_model_net3_ctrl_256_1000_2layers_2his_noval.pth',
        'flight_model_net4_ctrl_256_1000_2layers_2his_noval.pth',
        'flight_model_net5_ctrl_256_1000_2layers_2his_noval.pth']


class Net(nn.Module):
    def __init__(self, n_feature, n_hidden, n_output, n_layers):
        super(Net, self).__init__()
        self.bn_input = nn.BatchNorm1d(n_hidden, momentum=0.1)
        self.input = nn.Linear(n_feature, n_hidden)
        self.hiddens = nn.ModuleList()
        self.batchnorms = nn.ModuleList()
        for _ in range(n_layers-1):
            self.hiddens.append(nn.Linear(n_hidden, n_hidden))
            self.batchnorms.append(nn.BatchNorm1d(n_hidden, momentum=0.1))
            # self.hiddens.append(nn.Dropout(p=0.5))
        # self.hidden_1 = nn.Linear(n_feature, n_hidden)
        # self.hidden_2 = nn.Linear(n_hidden, n_hidden)
        self.predict = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x= self.input(x)
        x= self.bn_input(x)
        x= F.relu(x)
        for i in range(len(self.hiddens)):
            x = self.hiddens[i](x)
            x = self.batchnorms[i](x)
            x = F.relu(x)
            # x = F.dropout(x, p=0.5)
        # x= self.hidden_2(x)
        # x= self.bn_hidden(x)
        # x= F.relu(x)
        # x= F.dropout(x, p=0.5)
        x = self.predict(x)
        return x


class flight_dynamics(nn.Module):
    def __init__(self, T, n_batch):
        super(flight_dynamics, self).__init__()
        self.NUM_HISTORY = NUM_HISTORY
        self.NUM_ENSEMBLE = NUM_ENSEMBLE
        self.NUM_INPUTS = NUM_INPUTS
        self.NUM_HIDDEN_UNITS = NUM_HIDDEN_UNITS
        self.NUM_OUTPUTS = NUM_OUTPUTS
        self.NUM_HIDDEN_LAYERS = NUM_HIDDEN_LAYERS
        self.PATH = PATH
        self.goal_weights = torch.cat((torch.tensor([1., 1., 1., 0.5, 0.5, 0.5, 0.8, 0.8, 0.8]), torch.zeros(self.NUM_HISTORY * 13)))
        self.goal_state = torch.cat((torch.tensor([0., 0., 0., 0., 0., 0., 0., 0., 0.]), torch.zeros(self.NUM_HISTORY * 13)))
        self.ctrl_penalty = 0.001
        self.n_ctrl = 4
        self.n_state = self.NUM_INPUTS - self.n_ctrl
        self.lower = torch.tensor([-0.20, -0.1, -0.1, 0.6]).repeat(T, n_batch, 1)
        self.upper = torch.tensor([0.35, 0.1, 0.1, 0.8]).repeat(T, n_batch, 1)

        self.linesearch_decay = 0.2
        self.max_linesearch_iter = 5

        self.net = {}
        for i in range(self.NUM_ENSEMBLE):
            self.net['obj'+str(i)] = Net(self.NUM_INPUTS, self.NUM_HIDDEN_UNITS, self.NUM_OUTPUTS, self.NUM_HIDDEN_LAYERS)
            self.net['obj'+str(i)].load_state_dict(torch.load(self.PATH[i]))
            self.net['obj' + str(i)] = self.net['obj'+str(i)].eval()

    def forward(self, x, u):
        assert x.ndimension() == u.ndimension()
        if x.ndimension() == 1 and u.ndimension() ==1:
            x = x.unsqueeze(0)
            u = u.unsqueeze(0)
        exp_sum = torch.zeros(x.shape[0], int(self.NUM_OUTPUTS/2))
        xu = torch.cat((x[:, :(self.NUM_HISTORY + 1) * 9], u, x[:, (self.NUM_HISTORY + 1) * 9:]), dim=1)
        for i in range(self.NUM_ENSEMBLE):
            # net = Net(self.NUM_INPUTS, self.NUM_HIDDEN_UNITS, self.NUM_OUTPUTS, self.NUM_HIDDEN_LAYERS)
            # net.load_state_dict(torch.load(self.PATH[i]))
            # net = net.eval()
            prediction = self.net['obj' + str(i)](xu)
            exp, _ = torch.chunk(prediction, 2, dim=1)
            exp_sum += exp
        z = 1/self.NUM_ENSEMBLE * exp_sum
        z = torch.cat((z, x[:,:self.NUM_HISTORY*9], xu[:,(self.NUM_HISTORY + 1)*9:(self.NUM_HISTORY + 1)*9 + self.NUM_HISTORY * self.n_ctrl]), dim=1)
        return z

    def get_true_obj(self):
        q = torch.cat((
            self.goal_weights,
            self.ctrl_penalty * torch.ones(self.n_ctrl)
        ))
        assert not hasattr(self, 'mpc_lin')
        px = -torch.sqrt(self.goal_weights) * self.goal_state  # + self.mpc_lin
        p = torch.cat((px, torch.zeros(self.n_ctrl)))
        return q, p
