import torch
import numpy as np
# import bagread
# import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import time
from ..util import get_data_maybe


NUM_HISTORY = 2
NUM_INPUTS = (NUM_HISTORY + 1) * 13
NUM_OUTPUTS = 18
NUM_HIDDEN_UNITS = 200
NUM_HIDDEN_LAYERS = 2
# NUM_ENSEMBLE = 1
#
# PATH = ['flight_model_net1_ctrl_256_1000_2layers_2his_noval.pth',
#         'flight_model_net1_ctrl_256_1000_2layers_2his_noval.pth',
#         'flight_model_net3_ctrl_256_1000_2layers_2his_noval.pth',
#         'flight_model_net4_ctrl_256_1000_2layers_2his_noval.pth',
#         'flight_model_net5_ctrl_256_1000_2layers_2his_noval.pth']


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
        self.before_input_act = x
        x= F.relu(x)
        self.before_act = []
        for i in range(len(self.hiddens)):
            x = self.hiddens[i](x)
            x = self.batchnorms[i](x)
            self.before_act.append(x)
            x = F.relu(x)
            # x = F.dropout(x, p=0.5)
        # x= self.hidden_2(x)
        # x= self.bn_hidden(x)
        # x= F.relu(x)
        # x= F.dropout(x, p=0.5)
        x = self.predict(x)
        return x


class flight_dynamics(nn.Module):
    def __init__(self, T, n_batch, num_ensemble, path):
        super(flight_dynamics, self).__init__()
        self.NUM_HISTORY = NUM_HISTORY
        self.NUM_ENSEMBLE = num_ensemble
        self.NUM_INPUTS = NUM_INPUTS
        self.NUM_HIDDEN_UNITS = NUM_HIDDEN_UNITS
        self.NUM_OUTPUTS = NUM_OUTPUTS
        self.NUM_HIDDEN_LAYERS = NUM_HIDDEN_LAYERS
        self.PATH = path
        self.goal_weights = np.concatenate((np.array([10., 10., 10., 15., 10., 3., 2., 2., 2.], dtype='single'),
                                       # torch.tensor([10., 10., 10., 10., 10., 3., 2., 2., 2.]),
                                       # torch.tensor([10., 10., 10., 10., 10., 3., 2., 2., 2.]),
                                       np.zeros(self.NUM_HISTORY * 13, dtype='single')))
        self.goal_state = np.concatenate((np.array([0., 0., 0., -0.075, 0., 3.14, 0., 0., 0.], dtype='single'),
                                     # torch.tensor([0., 0., 0., -0.075, 0., 3.14, 0., 0., 0.]),
                                     # torch.tensor([0., 0., 0., -0.075, 0., 3.14, 0., 0., 0.]),
                                     np.zeros(self.NUM_HISTORY * 13, dtype='single')))
        self.ctrl_penalty = 20.
        self.goal_ctrl = np.array([0.075, 0., 0., 0.6], dtype='single')
        # self.slew_rate_penalty = torch.tensor([2., 2., 2., 2.])
        self.slew_rate_penalty = None
        self.n_ctrl = 4
        self.n_state = self.NUM_INPUTS - self.n_ctrl
        self.n_present_state = 9
        self.lower = np.tile(np.array([-0.05, -0.05, -0.05, 0.65], dtype='single'), (T, n_batch, 1))
        self.upper = np.tile(np.array([0.15, 0.05, 0.05, 0.75], dtype='single'), (T, n_batch, 1))
        # self.delta_u = torch.tensor(0.01)
        self.delta_u = None
        # self.lower = None
        # self.upper = None
        self.n_batch = n_batch

        self.linesearch_decay = 0.1
        self.max_linesearch_iter = 1

        self.net = {}
        for i in range(self.NUM_ENSEMBLE):
            self.net['obj'+str(i)] = Net(self.NUM_INPUTS, self.NUM_HIDDEN_UNITS, self.NUM_OUTPUTS, self.NUM_HIDDEN_LAYERS)
            self.net['obj'+str(i)].load_state_dict(torch.load(self.PATH[i]))
            self.net['obj' + str(i)] = self.net['obj'+str(i)].eval()

    def forward(self, x, u):
        # time1 = time.time()
        # assert x.ndimension() == u.ndimension()
        if x.ndim == 1 and u.ndim == 1:
            x = np.expand_dims(x, 0)
            u = np.expand_dims(u, 0)
        exp_sum = torch.zeros(x.shape[0], int(self.NUM_OUTPUTS/2))
        xu = np.concatenate((x[:, :(self.NUM_HISTORY + 1) * self.n_present_state], u, x[:, (self.NUM_HISTORY + 1) * self.n_present_state:]), axis=1)
        xu_torch = torch.from_numpy(xu)
        for i in range(self.NUM_ENSEMBLE):
            # net = Net(self.NUM_INPUTS, self.NUM_HIDDEN_UNITS, self.NUM_OUTPUTS, self.NUM_HIDDEN_LAYERS)
            # net.load_state_dict(torch.load(self.PATH[i]))
            # net = net.eval()
            prediction = self.net['obj' + str(i)](xu_torch)
            # print('Batch norm weight:', self.net['obj' + str(i)].predict.weight.size())
            exp, _ = torch.chunk(prediction, 2, dim=1)
            exp_sum += exp
        z = 1/self.NUM_ENSEMBLE * exp_sum
        z = np.concatenate((z.detach().numpy(), x[:,:self.NUM_HISTORY*self.n_present_state], xu[:,(self.NUM_HISTORY + 1)*self.n_present_state:(self.NUM_HISTORY + 1)*self.n_present_state + self.NUM_HISTORY * self.n_ctrl]), axis=1)
        # time2 = time.time()
        # print('model forward time:', time2 - time1)
        return z

    def get_true_obj(self):
        q = np.concatenate((
            self.goal_weights,
            self.ctrl_penalty * np.ones(self.n_ctrl, dtype='single')
        ))
        assert not hasattr(self, 'mpc_lin')
        # px = -torch.sqrt(self.goal_weights) * self.goal_state  # + self.mpc_lin
        px = -self.goal_weights * self.goal_state
        pu = -self.ctrl_penalty * np.ones(self.n_ctrl, dtype='single') * self.goal_ctrl
        p = np.concatenate((px, pu))
        return q, p


    def grad_input(self, x, u):
        n_batch_horizon = x.shape[0]
        for j in range(self.NUM_ENSEMBLE):
            grad = self.net['obj' + str(j)].predict.weight.repeat(n_batch_horizon,1,1).detach().numpy()
            for i in range(self.NUM_HIDDEN_LAYERS-2, -1, -1):
                I = get_data_maybe(self.net['obj' + str(j)].before_act[i] <= 0.).unsqueeze(2).repeat(1, 1, self.NUM_HIDDEN_UNITS)
                batchnorm_p = torch.div(self.net['obj' + str(j)].batchnorms[i].weight,
                                        torch.sqrt(self.net['obj' + str(j)].batchnorms[i].running_var) + 1e-5)
                Wi_grad = torch.mul(self.net['obj' + str(j)].hiddens[i].weight, batchnorm_p.reshape(-1,1)).repeat(n_batch_horizon,1,1)
                Wi_grad[I] = 0.
                # grad = grad.bmm(Wi_grad)
                grad = np.matmul(grad, Wi_grad.detach().numpy())
            I = get_data_maybe(self.net['obj' + str(j)].before_input_act <= 0.).unsqueeze(2).repeat(1, 1, self.NUM_INPUTS)
            batchnorm_p = torch.div(self.net['obj' + str(j)].bn_input.weight,
                                    torch.sqrt(self.net['obj' + str(j)].bn_input.running_var) + 1e-5)
            Wi_grad = torch.mul(self.net['obj' + str(j)].input.weight, batchnorm_p.reshape(-1,1)).repeat(n_batch_horizon, 1, 1)
            Wi_grad[I] = 0.
            # grad = grad.bmm(Wi_grad)
            grad = np.matmul(grad, Wi_grad.detach().numpy())
            if j == 0:
                grad_total = 1/self.NUM_ENSEMBLE * grad
            else:
                grad_total += grad
        R = np.concatenate((grad_total[:, :self.n_present_state, :(self.NUM_HISTORY + 1) * self.n_present_state],
                       grad_total[:, :self.n_present_state, self.NUM_HISTORY * self.n_present_state + 13:]), axis=2)
        S = grad_total[:, :self.n_present_state, (self.NUM_HISTORY + 1) * self.n_present_state:self.NUM_HISTORY * self.n_present_state + 13]
        if self.NUM_HISTORY >= 1:
            RHS = np.expand_dims(np.eye(self.NUM_HISTORY * self.n_present_state, self.n_state, dtype='single'), 0).repeat(n_batch_horizon, 0)
            # print('RHS: {}'. format(RHS[0,:,:]))
            SHS = np.expand_dims(np.zeros((self.NUM_HISTORY * self.n_present_state, self.n_ctrl), dtype='single'), 0).repeat(n_batch_horizon, 0)
            # print('SHS: {}'.format(SHS[0,:,:]))
            RU = np.expand_dims(np.zeros((self.n_ctrl, self.n_state), dtype='single'), 0).repeat(n_batch_horizon, 0)
            # print('RU: {}'.format(RU[0,:,:]))
            SU = np.expand_dims(np.eye(self.n_ctrl, dtype='single'), 0).repeat(n_batch_horizon, 0)
            # print('SU: {}'.format(SU[0,:,:]))
            RHU = np.expand_dims(np.concatenate((np.zeros(((self.NUM_HISTORY - 1) * self.n_ctrl, (self.NUM_HISTORY + 1) * self.n_present_state), dtype='single'),
                             np.eye((self.NUM_HISTORY - 1) * self.n_ctrl, self.NUM_HISTORY * self.n_ctrl, dtype='single')), axis=1), 0).repeat(n_batch_horizon, 0)
            # print('RHu: {}'.format(RHU[0,:,:]))
            SHU = np.expand_dims(np.zeros(((self.NUM_HISTORY - 1) * self.n_ctrl, self.n_ctrl), dtype='single'), 0).repeat(n_batch_horizon, 0)
            # print('SHU: {}'.format(SHU[0,:,:]))
            R = np.concatenate((R, RHS, RU, RHU), axis=1)
            # print('R: {}'.format(R[0,:,:]))
            S = np.concatenate((S, SHS, SU, SHU), axis=1)
            # print('S: {}'.format(S[0,:,:]))
        else:
            pass
        return R, S