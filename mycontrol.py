#!/usr/bin/env python3

import argparse

import torch
from torch.autograd import Function, Variable
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter
import time
import numpy as np

import os
from collections import namedtuple
from mpc import mpc, util
from mpc.mpc import GradMethods
import mpc.util as eutil
from mpc.env_dx import pendulum, cartpole, Gemini_flight_dynamics

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.style.use('bmh')

P = torch.tensor([0.06, -0.18, 0.09, 0.52])
I = torch.tensor([0.1, -0.27, 0.4, 0.4])
D = torch.tensor([0.001, -0.01, 0.01, 0.04])
dt = 0.02
NUM_ENSEMBLE_CONTROL = 1
NUM_ENSEMBLE_PREDICT = 1
PATH_CONTROL = ['flight_model_net1_ctrl_256_1000_2layers_2his_noval.pth',
                'flight_model_net2_ctrl_256_1000_2layers_2his_noval.pth',
                'flight_model_net3_ctrl_256_1000_2layers_2his_noval.pth',
                'flight_model_net4_ctrl_256_1000_2layers_2his_noval.pth',
                'flight_model_net5_ctrl_256_1000_2layers_2his_noval.pth']

PATH_PREDICT = ['flight_model_net2_ctrl_256_1000_2layers_2his_noval.pth',
                'flight_model_net2_ctrl_256_1000_2layers_2his_noval.pth',
                'flight_model_net3_ctrl_256_1000_2layers_2his_noval.pth',
                'flight_model_net4_ctrl_256_1000_2layers_2his_noval.pth',
                'flight_model_net5_ctrl_256_1000_2layers_2his_noval.pth']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Gemini_flight_dynamics')
    args = parser.parse_args()

    torch.manual_seed(1)
    n_batch = 1
    if args.env == 'pendulum':
        T = 5
        params = torch.tensor((10., 1., 1.))  # Gravity, mass, length.
        START = time.time()
        dx = pendulum.PendulumDx(params, simple=True)
        END = time.time()
        print('initialize model time:', END - START)
        xinit = torch.zeros(n_batch, dx.n_state)
        th = 1.0
        xinit[:,0] = np.cos(th)
        xinit[:,1] = np.sin(th)
        xinit[:,2] = -0.5
    elif args.env == 'cartpole':
        T = 20
        dx = cartpole.CartpoleDx()
        xinit = torch.zeros(n_batch, dx.n_state)
        th = 0.5
        xinit[:,2] = np.cos(th)
        xinit[:,3] = np.sin(th)
    elif args.env == 'Gemini_flight_dynamics':
        T = 5
        # START = time.time()
        dx_control = Gemini_flight_dynamics.flight_dynamics(T, n_batch, NUM_ENSEMBLE_CONTROL, PATH_CONTROL)
        dx_predict = Gemini_flight_dynamics.flight_dynamics(T, n_batch, NUM_ENSEMBLE_PREDICT, PATH_PREDICT)
        # END = time.time()
        # print('initialize model time:', END - START)
        xinit = torch.tensor([-5.8489e-03,  4.3651e-02, -7.9248e-02, -8.5625e-02,  1.1335e-02,
         3.1228e+00,  1.0082e-02, -6.1053e-02,  1.4188e-02, -7.4810e-03,
         4.2597e-02, -7.3945e-02, -8.6181e-02,  1.2132e-02,  3.1227e+00,
         1.2712e-02, -6.5020e-02,  1.2549e-02, -1.0209e-02,  4.1048e-02,
        -6.6239e-02, -8.6811e-02,  1.3238e-02,  3.1226e+00, -7.5656e-03,
        -5.4819e-02,  1.9917e-04,  1.3622e-01,  2.4634e-03,  1.0794e-03,
         6.6767e-01,  1.3339e-01,  2.5716e-03,  3.0723e-03,  6.6944e-01]).unsqueeze(0).repeat(n_batch,1)

    else:
        assert False

    q, p = dx_control.get_true_obj()

    u = dx_control.goal_ctrl.repeat(T, n_batch, 1)
    # u= None
    ep_length = 100
    x_plot = []
    u_plot = []
    # error_int = torch.zeros(T, n_batch, dx_control.n_ctrl)
    # prev_error = torch.zeros(T, n_batch, dx_control.n_ctrl)
    for t in range(ep_length):
        start_ilqr = time.time()
        x, u = solve_lqr(
            dx_control, xinit, q, p, T, dx_control.linesearch_decay, dx_control.max_linesearch_iter, u)
        end_ilqr = time.time()
        print('one step MPC:', end_ilqr - start_ilqr)
        # print('epoch:', t, '| u:', u[0])
        x_plot.append(x.detach().numpy())
        u_plot.append(u.detach().numpy())
        # error = torch.cat((dx_control.goal_state[6:9], dx_control.goal_state[2:3])).repeat(T, n_batch, 1) - torch.cat((x[:,:,6:9], x[:,:,2:3]), dim=2)
        # error_int = util.eclamp(error_int + I * error * dt, -torch.tensor([0.5, 0.5, 0.5, 0.5]).repeat(T, n_batch, 1), torch.tensor([0.5, 0.5, 0.5, 0.5]).repeat(T, n_batch, 1))
        # u = util.eclamp(P * error + error_int + D * (error - prev_error) / dt + dx_control.goal_ctrl.repeat(T, n_batch, 1), dx_control.lower, dx_control.upper)
        # prev_error = error
        # print('initial u:', u)
        xinit = dx_predict(x[0], u[0])
        # u = torch.cat((u[1:-1], u[-2:]), 0).contiguous()
        # u = dx_control.goal_ctrl.repeat(T, n_batch, 1)
        # u = u[0].repeat(T, n_batch, 1)
        # xinit = x[1]
    trajectory_plot(x_plot, T)
    action_plot(u_plot, T)
    plt.show()


def solve_lqr(dx, xinit, q, p, T,
              linesearch_decay, max_linesearch_iter, u_init=None):
    n_sc = dx.n_state+dx.n_ctrl

    n_batch = 1
    Q = torch.diag(q).unsqueeze(0).unsqueeze(0).repeat(
        T, n_batch, 1, 1
    )
    p = p.unsqueeze(0).repeat(T, n_batch, 1)

    # print(QuadCost(Q,p))
    lqr_iter = 1 if u_init is None else 1
    x_lqr, u_lqr, objs_lqr = mpc.MPC(
        dx.n_state, dx.n_ctrl, T,
        u_lower=dx.lower,
        u_upper=dx.upper,
        delta_u=dx.delta_u,
        u_init=u_init,
        lqr_iter=lqr_iter,
        verbose=0,
        exit_unconverged=False,
        detach_unconverged=False,
        linesearch_decay=linesearch_decay,
        max_linesearch_iter=max_linesearch_iter,
        slew_rate_penalty= dx.slew_rate_penalty,
        grad_method=GradMethods.ANALYTIC,
        eps=1e-5,
        # slew_rate_penalty=self.slew_rate_penalty,
        # prev_ctrl=prev_ctrl,
    )(xinit, mpc.QuadCost(Q, p), dx)
    return x_lqr, u_lqr


def trajectory_plot(x_plot, T):
    plt.figure(1, figsize=(10,10))
    titles = ("velocity_body_x", "velocity_body_y", "velocity_body_z", "roll", "pitch", "yaw", "angular_velocity_x",
              "angular_velocity_y", "angular_velocity_z")
    for i in range(9):
        plt.subplot(3,3,i+1)
        for j in range(len(x_plot)):
            step = np.arange(T)+j
            plt.plot(step, x_plot[j][:, 0, i])
        plt.title(titles[i])
        # plt.legend(loc='upper right')
    # plt.show()


def action_plot(u_plot, T):
    plt.figure(2, figsize=(10, 10))
    titles = ("actuator_roll", "actuator_pitch", "actuator_yaw", "actuator_z")
    for i in range(4):
        plt.subplot(4,1,i+1)
        # step = range(len(u_plot))
        # u = [u_plot[j][0,0,i] for j in range(len(u_plot))]
        # plt.plot(step, u)
        for j in range(len(u_plot)):
            step = np.arange(T) + j
            plt.plot(step, u_plot[j][:, 0, i])
        plt.title(titles[i])

if __name__ == '__main__':
    main()