#!/usr/bin/env python3

import argparse

import torch
# from torch.autograd import Function, Variable
# import torch.nn.functional as F
# from torch import nn
# from torch.nn.parameter import Parameter
import time
import numpy as np

import os
from collections import namedtuple
from mpc import mpc, util
from mpc.mpc import GradMethods
# import mpc.util as eutil
from mpc.env_dx import pendulum, cartpole, Gemini_flight_dynamics

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.style.use('bmh')

P = np.array([0.06, -0.18, 0.09, 0.52], dtype='single')
I = np.array([0.1, -0.27, 0.4, 0.4], dtype='single')
D = np.array([0.001, -0.01, 0.01, 0.04], dtype='single')
dt = 0.02
NUM_ENSEMBLE_CONTROL = 1
NUM_ENSEMBLE_PREDICT = 1
PATH_CONTROL = ['pathfiles/flight_model_net1_ctrl_256_1000_2layers_2his_noval.pth',
                'pathfiles/flight_model_net2_ctrl_256_1000_2layers_2his_noval.pth',
                'pathfiles/flight_model_net3_ctrl_256_1000_2layers_2his_noval.pth',
                'pathfiles/flight_model_net4_ctrl_256_1000_2layers_2his_noval.pth',
                'pathfiles/flight_model_net5_ctrl_256_1000_2layers_2his_noval.pth']

PATH_PREDICT = ['pathfiles/flight_model_net1_ctrl_256_1000_2layers_2his_noval.pth',
                'pathfiles/flight_model_net2_ctrl_256_1000_2layers_2his_noval.pth',
                'pathfiles/flight_model_net3_ctrl_256_1000_2layers_2his_noval.pth',
                'pathfiles/flight_model_net4_ctrl_256_1000_2layers_2his_noval.pth',
                'pathfiles/flight_model_net5_ctrl_256_1000_2layers_2his_noval.pth']


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
        xinit = np.expand_dims(np.array([ 3.0953e-03,  3.8064e-03,  4.1662e-03, -7.4164e-02,  2.0541e-02,
         3.1526e+00, -2.7789e-02,  4.7386e-02, -3.4559e-02,  5.2193e-04,
         3.8532e-03,  3.3118e-03, -7.3201e-02,  2.0192e-02,  3.1536e+00,
        -1.0387e-01, -1.9162e-02, -5.6059e-02, -2.5459e-03,  5.3939e-03,
         1.8183e-03, -7.1328e-02,  2.0130e-02,  3.1550e+00, -4.6090e-02,
         5.5544e-02, -7.8294e-02,  6.0124e-02, -2.6421e-02,  8.4291e-03,
         6.8993e-01,  7.2849e-02, -1.3616e-02,  1.0076e-02,  6.9233e-01], dtype='single'), axis=0).repeat(n_batch, axis=0)

    else:
        assert False

    q, p = dx_control.get_true_obj()

    u = np.tile(dx_control.goal_ctrl, (T, n_batch, 1))
    # u= None
    ep_length = 100
    x_plot = []
    u_plot = []
    MPC_time = 0.
    # error_int = torch.zeros(T, n_batch, dx_control.n_ctrl)
    # prev_error = torch.zeros(T, n_batch, dx_control.n_ctrl)
    for t in range(ep_length):
        start_ilqr = time.time()
        x, u = solve_lqr(
            dx_control, xinit, q, p, T, dx_control.linesearch_decay, dx_control.max_linesearch_iter, u)
        end_ilqr = time.time()
        print('one step MPC:', end_ilqr - start_ilqr)
        MPC_time += end_ilqr - start_ilqr
        # print('epoch:', t, '| u:', u[0])
        x_plot.append(x)
        u_plot.append(u)
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
    print('average mpc one step time:', MPC_time/ep_length)
    # trajectory_plot(x_plot, T)
    # action_plot(u_plot, T)
    # plt.show()


def solve_lqr(dx, xinit, q, p, T,
              linesearch_decay, max_linesearch_iter, u_init=None):
    n_sc = dx.n_state+dx.n_ctrl

    n_batch = 1
    Q = np.tile(np.diag(q), (T, n_batch, 1, 1))
    p = np.tile(p, (T, n_batch, 1))

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