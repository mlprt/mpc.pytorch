import torch
from torch.autograd import Function, Variable
from torch.nn import Module
from torch.nn.parameter import Parameter

import numpy as np
import numpy.random as npr

from collections import namedtuple

import time

from . import util, mpc
from .pnqp import pnqp

LqrBackOut = namedtuple('lqrBackOut', 'n_total_qp_iter')
# LqrForOut = namedtuple(
#     'lqrForOut',
#     'objs objsxx objsuu objsx objsu full_du_norm alpha_du_norm mean_alphas costs costsxx costsuu costsx costsu'
# )
LqrForOut = namedtuple(
    'lqrForOut',
    'objs full_du_norm alpha_du_norm mean_alphas costs'
)
# @staticmethod
class LQRStep(Module):
    """A single step of the box-constrained iLQR solver.

    Required Args:
        n_state, n_ctrl, T
        x_init: The initial state [n_batch, n_state]

    Optional Args:
        u_lower, u_upper: The lower- and upper-bounds on the controls.
            These can either be floats or shaped as [T, n_batch, n_ctrl]
            TODO: Better support automatic expansion of these.
        TODO
    """

    def __init__(
            self,
            n_state,
            n_ctrl,
            T,
            u_lower=None,
            u_upper=None,
            u_zero_I=None,
            delta_u=None,
            linesearch_decay=0.2,
            max_linesearch_iter=1,
            true_cost=None,
            true_dynamics=None,
            delta_space=True,
            current_x=None,
            current_u=None,
            verbose=0,
            back_eps=1e-3,
            no_op_forward=False,
    ):
        super(LQRStep, self).__init__()
        self.n_state = n_state
        self.n_ctrl = n_ctrl
        self.T = T

        self.u_lower = util.get_data_maybe(u_lower)
        self.u_upper = util.get_data_maybe(u_upper)

        # TODO: Better checks for this
        if isinstance(self.u_lower, int):
            self.u_lower = float(self.u_lower)
        if isinstance(self.u_upper, int):
            self.u_upper = float(self.u_upper)
        if isinstance(self.u_lower, np.float32):
            self.u_lower = u_lower.item()
        if isinstance(self.u_upper, np.float32):
            self.u_upper = u_upper.item()

        self.u_zero_I = u_zero_I
        self.delta_u = delta_u
        self.linesearch_decay = linesearch_decay
        self.max_linesearch_iter = max_linesearch_iter
        self.true_cost = true_cost
        self.true_dynamics = true_dynamics
        self.delta_space = delta_space
        self.current_x = util.get_data_maybe(current_x)
        self.current_u = util.get_data_maybe(current_u)
        self.verbose = verbose

        self.back_eps = back_eps

        self.no_op_forward = no_op_forward

    # @profile
    # @staticmethod
    def forward(self, x_init, C, c, F, f=None):
        if self.no_op_forward:
            # self.save_for_backward(
            #     x_init, C, c, F, f, self.current_x, self.current_u)
            return self.current_x, self.current_u

        if self.delta_space:
            # Taylor-expand the objective to do the backward pass in
            # the delta space.
            assert self.current_x is not None
            assert self.current_u is not None
            c_back = []
            for t in range(self.T):
                xt = self.current_x[t]
                ut = self.current_u[t]
                xut = np.concatenate((xt, ut), 1)
                c_back.append(util.bmv(C[t], xut) + c[t])
            c_back = np.stack(c_back)
            f_back = None
        else:
            assert False

        # time1 = time.time()
        Ks, ks, self.back_out = self.lqr_backward(C, c_back, F, f_back)
        # time2 = time.time()
        # time3 = time.time()
        new_x, new_u, self.for_out = self.lqr_forward(
            x_init, C, c, F, f, Ks, ks)
        # time4 = time.time()
        # print('lqr forward time:', time4 - time3, 'lqr backward time:', time2 -time1)
        # self.save_for_backward(x_init, C, c, F, f, new_x, new_u)
        return new_x, new_u
    
    # @staticmethod
    # def backward(self, dl_dx, dl_du):
    #     start = time.time()
    #     # x_init, C, c, F, f, new_x, new_u = self.saved_tensors
    #
    #     r = []
    #     for t in range(self.T):
    #         rt = torch.cat((dl_dx[t], dl_du[t]), 1)
    #         r.append(rt)
    #     r = torch.stack(r)
    #
    #     if self.u_lower is None:
    #         I = None
    #     else:
    #         I = (torch.abs(new_u - self.u_lower) <= 1e-8) | \
    #             (torch.abs(new_u - self.u_upper) <= 1e-8)
    #     dx_init = Variable(torch.zeros_like(x_init))
    #     _mpc = mpc.MPC(
    #         self.n_state, self.n_ctrl, self.T,
    #         u_zero_I=I,
    #         u_init=None,
    #         lqr_iter=1,
    #         verbose=-1,
    #         n_batch=C.size(1),
    #         delta_u=None,
    #         # exit_unconverged=True, # It's really bad if this doesn't converge.
    #         exit_unconverged=False, # It's really bad if this doesn't converge.
    #         eps=self.back_eps,
    #     )
    #     dx, du, _ = _mpc(dx_init, mpc.QuadCost(C, -r), mpc.LinDx(F, None))
    #
    #     dx, du = dx.data, du.data
    #     dxu = torch.cat((dx, du), 2)
    #     xu = torch.cat((new_x, new_u), 2)
    #
    #     dC = torch.zeros_like(C)
    #     for t in range(self.T):
    #         xut = torch.cat((new_x[t], new_u[t]), 1)
    #         dxut = dxu[t]
    #         dCt = -0.5*(util.bger(dxut, xut) + util.bger(xut, dxut))
    #         dC[t] = dCt
    #
    #     dc = -dxu
    #
    #     lams = []
    #     prev_lam = None
    #     for t in range(self.T-1, -1, -1):
    #         Ct_xx = C[t,:,:self.n_state,:self.n_state]
    #         Ct_xu = C[t,:,:self.n_state,self.n_state:]
    #         ct_x = c[t,:,:self.n_state]
    #         xt = new_x[t]
    #         ut = new_u[t]
    #         lamt = util.bmv(Ct_xx, xt) + util.bmv(Ct_xu, ut) + ct_x
    #         if prev_lam is not None:
    #             Fxt = F[t,:,:,:self.n_state].transpose(1, 2)
    #             lamt += util.bmv(Fxt, prev_lam)
    #         lams.append(lamt)
    #         prev_lam = lamt
    #     lams = list(reversed(lams))
    #
    #     dlams = []
    #     prev_dlam = None
    #     for t in range(self.T-1, -1, -1):
    #         dCt_xx = C[t,:,:self.n_state,:self.n_state]
    #         dCt_xu = C[t,:,:self.n_state,self.n_state:]
    #         drt_x = -r[t,:,:self.n_state]
    #         dxt = dx[t]
    #         dut = du[t]
    #         dlamt = util.bmv(dCt_xx, dxt) + util.bmv(dCt_xu, dut) + drt_x
    #         if prev_dlam is not None:
    #             Fxt = F[t,:,:,:self.n_state].transpose(1, 2)
    #             dlamt += util.bmv(Fxt, prev_dlam)
    #         dlams.append(dlamt)
    #         prev_dlam = dlamt
    #     dlams = torch.stack(list(reversed(dlams)))
    #
    #     dF = torch.zeros_like(F)
    #     for t in range(self.T-1):
    #         xut = xu[t]
    #         lamt = lams[t+1]
    #
    #         dxut = dxu[t]
    #         dlamt = dlams[t+1]
    #
    #         dF[t] = -(util.bger(dlamt, xut) + util.bger(lamt, dxut))
    #
    #     if f.nelement() > 0:
    #         _dlams = dlams[1:]
    #         assert _dlams.shape == f.shape
    #         df = -_dlams
    #     else:
    #         df = torch.Tensor()
    #
    #     dx_init = -dlams[0]
    #
    #     self.backward_time = time.time()-start
    #     return dx_init, dC, dc, dF, df

    # @profile
    def lqr_backward(self, C, c, F, f):
        n_batch = C.shape[1]

        u = self.current_u
        Ks = []
        ks = []
        prev_kt = None
        n_total_qp_iter = 0
        Vtp1 = vtp1 = None
        for t in range(self.T-1, -1, -1):
            if t == self.T-1:
                Qt = C[t]
                qt = c[t]
            else:
                Ft = F[t]
                Ft_T = Ft.transpose(0,2,1)
                Qt = C[t] + np.matmul(np.matmul(Ft_T, Vtp1), Ft)
                if f is None or f.size == 0:
                    qt = c[t] + np.matmul(Ft_T, np.expand_dims(vtp1, 2)).squeeze()
                else:
                    ft = f[t]
                    qt = c[t] + np.matmul(np.matmul(Ft_T, Vtp1), np.expand_dims(ft, 2)).squeeze() + \
                        np.matmul(Ft_T, np.expand_dims(vtp1, 2)).squeeze()

            n_state = self.n_state
            Qt_xx = Qt[:, :n_state, :n_state]
            Qt_xu = Qt[:, :n_state, n_state:]
            Qt_ux = Qt[:, n_state:, :n_state]
            Qt_uu = Qt[:, n_state:, n_state:]
            qt_x = qt[:, :n_state]
            qt_u = qt[:, n_state:]

            if self.u_lower is None:
                if self.n_ctrl == 1 and self.u_zero_I is None:
                    Kt = -(1./Qt_uu)*Qt_ux
                    kt = -(1./Qt_uu.squeeze(2))*qt_u
                else:
                    if self.u_zero_I is None:
                        Qt_uu_inv = np.linalg.inv(Qt_uu)
                        Kt = np.matmul(-Qt_uu_inv, Qt_ux)
                        kt = util.bmv(-Qt_uu_inv, qt_u)

                        # Qt_uu_LU = Qt_uu.lu()
                        # Kt = -Qt_ux.lu_solve(*Qt_uu_LU)
                        # kt = -qt_u.lu_solve(*Qt_uu_LU)
                    else:
                        # Solve with zero constraints on the active controls.
                        I = self.u_zero_I[t].float()
                        notI = 1-I

                        qt_u_ = qt_u.copy()
                        qt_u_[I.bool()] = 0

                        Qt_uu_ = Qt_uu.copy()

                        if I.is_cuda:
                            notI_ = notI.float()
                            Qt_uu_I = (1-util.bger(notI_, notI_)).type_as(I)
                        else:
                            Qt_uu_I = 1-util.bger(notI, notI)

                        Qt_uu_[Qt_uu_I.bool()] = 0.
                        Qt_uu_[util.bdiag(I).bool()] += 1e-8

                        Qt_ux_ = Qt_ux.copy()
                        Qt_ux_[np.expand_dims(I, 2).repeat(Qt_ux.shape[2], 2).bool()] = 0.

                        if self.n_ctrl == 1:
                            Kt = -(1./Qt_uu_)*Qt_ux_
                            kt = -(1./Qt_uu.squeeze(2))*qt_u_
                        else:
                            Qt_uu_LU_ = Qt_uu_.lu()
                            Kt = -Qt_ux_.lu_solve(*Qt_uu_LU_)
                            kt = -qt_u_.unsqueeze(2).lu_solve(*Qt_uu_LU_).squeeze(2)
            # else:
            #     assert self.delta_space
            #     lb = self.get_bound('lower', t) - u[t]
            #     ub = self.get_bound('upper', t) - u[t]
            #     if self.delta_u is not None:
            #         lb[lb < -self.delta_u] = -self.delta_u
            #         ub[ub > self.delta_u] = self.delta_u
            #     kt, Qt_uu_free_LU, If, n_qp_iter = pnqp(
            #         Qt_uu, qt_u, lb, ub,
            #         x_init=prev_kt, n_iter=10)
            #     if self.verbose > 1:
            #         print('  + n_qp_iter: ', n_qp_iter+1)
            #     n_total_qp_iter += 1+n_qp_iter
            #     prev_kt = kt
            #     Qt_ux_ = Qt_ux.clone()
            #     Qt_ux_[(1-If).unsqueeze(2).repeat(1,1,Qt_ux.size(2)).bool()] = 0
            #     if self.n_ctrl == 1:
            #         # Bad naming, Qt_uu_free_LU isn't the LU in this case.
            #         Kt = -((1./Qt_uu_free_LU)*Qt_ux_)
            #     else:
            #         Kt = -Qt_ux_.lu_solve(*Qt_uu_free_LU)
            else:
                Qt_uu_inv = np.linalg.inv(Qt_uu)
                # print(Qt_uu_inv.dtype)
                Kt = np.matmul(-Qt_uu_inv, Qt_ux)
                kt = util.bmv(-Qt_uu_inv, qt_u)
            Kt_T = Kt.transpose(0,2,1)

            Ks.append(Kt)
            ks.append(kt)

            Vtp1 = Qt_xx + np.matmul(Qt_xu, Kt) + np.matmul(Kt_T, Qt_ux) + np.matmul(np.matmul(Kt_T, Qt_uu), Kt)
            vtp1 = qt_x + np.matmul(Qt_xu, np.expand_dims(kt, 2)).squeeze(2) + \
                np.matmul(Kt_T, np.expand_dims(qt_u, 2)).squeeze(2) + \
                np.matmul(np.matmul(Kt_T, Qt_uu), np.expand_dims(kt, 2)).squeeze(2)

        return Ks, ks, LqrBackOut(n_total_qp_iter=n_total_qp_iter)


    # @profile
    def lqr_forward(self, x_init, C, c, F, f, Ks, ks):
        # print(Ks[0].dtype, Ks[1].dtype, Ks[2].dtype, Ks[3].dtype, Ks[4].dtype)
        x = self.current_x
        u = self.current_u
        n_batch = C.shape[1]

        old_cost = util.get_cost(self.T, u, self.true_cost, self.true_dynamics, x=x)

        current_cost = None
        alphas = np.ones(n_batch, dtype='single')
        full_du_norm = None

        i = 0
        while (current_cost is None or \
               (old_cost is not None and \
                  np.any((current_cost > old_cost)).item() == 1)) and \
              i < self.max_linesearch_iter:
            new_u = []
            new_x = [x_init]
            dx = [np.zeros_like(x_init)]
            objs = []
            objsxx = []
            objsuu = []
            objsx = []
            objsu = []
            for t in range(self.T):
                t_rev = self.T-1-t
                Kt = Ks[t_rev]
                kt = ks[t_rev]
                new_xt = new_x[t]
                xt = x[t]
                ut = u[t]
                dxt = dx[t]
                new_ut = util.bmv(Kt, dxt) + ut + np.matmul(np.diag(alphas), kt)
                # new_ut = util.bmv(Kt, dxt) + ut + kt

                # Currently unimplemented:
                assert not ((self.delta_u is not None) and (self.u_lower is None))

                if self.u_zero_I is not None:
                    new_ut[self.u_zero_I[t]] = 0.

                if self.u_lower is not None:
                    lb = self.get_bound('lower', t)
                    ub = self.get_bound('upper', t)

                    if self.delta_u is not None:
                        lb_limit, ub_limit = lb, ub
                        lb = u[t] - self.delta_u
                        ub = u[t] + self.delta_u
                        I = lb < lb_limit
                        lb[I] = lb_limit if isinstance(lb_limit, float) else lb_limit[I]
                        I = ub > ub_limit
                        ub[I] = ub_limit if isinstance(lb_limit, float) else ub_limit[I]
                    new_ut = util.eclamp(new_ut, lb, ub)
                new_u.append(new_ut)

                new_xut = np.concatenate((new_xt, new_ut), axis=1)
                if t < self.T-1:
                    if isinstance(self.true_dynamics, mpc.LinDx):
                        F, f = self.true_dynamics.F, self.true_dynamics.f
                        new_xtp1 = util.bmv(F[t], new_xut)
                        if f is not None and f.nelement() > 0:
                            new_xtp1 += f[t]
                    else:
                        # print(new_xt.dtype, new_ut.dtype)
                        new_xtp1 = self.true_dynamics(
                            new_xt, new_ut)

                    new_x.append(new_xtp1)
                    dx.append(new_xtp1 - x[t+1])

                if isinstance(self.true_cost, mpc.QuadCost):
                    C, c = self.true_cost.C, self.true_cost.c
                    if self.verbose > 0:
                        Cxx = C[:, :, :self.n_state, :self.n_state]
                        Cuu = C[:, :, self.n_state:, self.n_state:]
                        cx = c[:, :, :self.n_state]
                        cu = c[:, :, self.n_state:]
                        objxx = 0.5*util.bquad(new_xt - np.expand_dims(self.true_dynamics.goal_state, 0), Cxx[t])
                        objuu = 0.5*util.bquad(new_ut - np.expand_dims(self.true_dynamics.goal_ctrl, 0), Cuu[t])
                        # objx = util.bdot(new_xt, cx[t])
                        # obju = util.bdot(new_ut, cu[t])
                    # obj = 0.5*util.bquad(new_xut, C[t]) + util.bdot(new_xut, c[t]) + \
                    #       0.5*util.bquad(torch.cat((self.true_dynamics.goal_state.repeat(1,1), self.true_dynamics.goal_ctrl.repeat(1,1)), dim=1), C[t])
                    obj = 0.5*util.bquad(new_xut - np.concatenate((np.expand_dims(self.true_dynamics.goal_state, 0), np.expand_dims(self.true_dynamics.goal_ctrl, 0)), axis=1), C[t])
                else:
                    obj = self.true_cost(new_xut)

                if self.verbose > 0:
                    objsxx.append(objxx)
                    objsuu.append(objuu)
                    # objsx.append(objx)
                    # objsu.append(obju)
                objs.append(obj)
            if self.verbose > 0:
                objsxx = np.stack(objsxx)
                objsuu = np.stack(objsuu)
                # objsx = torch.stack(objsx)
                # objsu = torch.stack(objsu)
                current_costxx = np.sum(objsxx, axis=0)
                current_costuu = np.sum(objsuu, axis=0)
                # current_costx = torch.sum(objsx, dim=0)
                # current_costu = torch.sum(objsu, dim=0)
            objs = np.stack(objs)
            current_cost = np.sum(objs, axis=0)
            new_u = np.stack(new_u)
            new_x = np.stack(new_x)
            if full_du_norm is None:
                full_du_norm = np.linalg.norm((u-new_u).transpose(0,2,1).reshape(
                    n_batch, -1), ord=2, axis=1)

            alphas[current_cost > old_cost] *= self.linesearch_decay
            i += 1

        # If the iteration limit is hit, some alphas
        # are one step too small.
        alphas[current_cost > old_cost] /= self.linesearch_decay
        alpha_du_norm = np.linalg.norm((u-new_u).transpose(0,2,1).reshape(
            n_batch, -1), ord=2, axis=1)

        return new_x, new_u, LqrForOut(
            objs,
            full_du_norm,
            alpha_du_norm,
            np.mean(alphas),
            current_cost,
        )


    def get_bound(self, side, t):
        v = getattr(self, 'u_'+side)
        if isinstance(v, float):
            return v
        else:
            return v[t]
