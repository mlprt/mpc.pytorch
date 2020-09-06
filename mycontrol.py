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

# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# plt.style.use('bmh')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Gemini_flight_dynamics')
    args = parser.parse_args()

    torch.manual_seed(1)
    n_batch = 1
    if args.env == 'pendulum':
        T = 5
        params = torch.tensor((10., 1., 1.))  # Gravity, mass, length.
        dx = pendulum.PendulumDx(params, simple=True)
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
        dx = Gemini_flight_dynamics.flight_dynamics(T, n_batch)
        xinit = torch.zeros(n_batch, dx.n_state)
    else:
        assert False

    q, p = dx.get_true_obj()

    u = None
    ep_length = 1
    for t in range(ep_length):
        start_ilqr = time.time()
        x, u = solve_lqr(
            dx, xinit, q, p, T, dx.linesearch_decay, dx.max_linesearch_iter, u)
        end_ilqr = time.time()
        print('one step MPC:', end_ilqr - start_ilqr)

        u = torch.cat((u[1:-1], u[-2:]), 0).contiguous()
        xinit = x[1]


def solve_lqr(dx, xinit, q, p, T,
              linesearch_decay, max_linesearch_iter, u_init=None):
    n_sc = dx.n_state+dx.n_ctrl

    n_batch = 1
    Q = torch.diag(q).unsqueeze(0).unsqueeze(0).repeat(
        T, n_batch, 1, 1
    )
    p = p.unsqueeze(0).repeat(T, n_batch, 1)

    # print(QuadCost(Q,p))
    lqr_iter = 10 if u_init is None else 10
    x_lqr, u_lqr, objs_lqr = mpc.MPC(
        dx.n_state, dx.n_ctrl, T,
        u_lower=dx.lower,
        u_upper=dx.upper,
        u_init=u_init,
        lqr_iter=lqr_iter,
        verbose=0,
        exit_unconverged=False,
        detach_unconverged=False,
        linesearch_decay=linesearch_decay,
        max_linesearch_iter=max_linesearch_iter,
        grad_method=GradMethods.FINITE_DIFF,
        eps=1e-4,
        # slew_rate_penalty=self.slew_rate_penalty,
        # prev_ctrl=prev_ctrl,
    )(xinit, mpc.QuadCost(Q, p), dx)
    return x_lqr, u_lqr


if __name__ == '__main__':
    main()