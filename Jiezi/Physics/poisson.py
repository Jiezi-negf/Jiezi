# Copyright 2021 Jiezi authors.
#
# This file is part of Jiezi. It is subject to the license terms in the file
# LICENSE.md found in the top-level directory of this distribution. A list of
# Jiezi authors can be found in the file AUTHORS.md at the top-level directory
# of this distribution.
# ==============================================================================

import numpy as np
import math
from dolfin import *
import matplotlib.pyplot as plt
import copy
from Jiezi.FEM.constant_parameters import constant_parameters
from Jiezi.FEM.map import map_tocell
from Jiezi.FEM.ef_solver import ef_solver
from Jiezi.FEM.assembly import assembly_nonlinear, assembly_linear
from Jiezi.Physics.common import time_it


@ time_it
def poisson(mode, info_mesh, N_GP_T, cell_long_term, cell_NJ, cell_NNTJ, cnt_cell_list,
            ef_init_n, ef_init_p,
            Dirichlet_list, Dirichlet_value, E_list, Ec, Eg, TOL_ef, TOL_du,
            dos_GP_list, n_GP_list, p_GP_list, doping_GP_list, u_init, dof_amount):
    print("poisson solver start")
    phi = None
    if mode == 1:
        phi = poisson_nonlinear(info_mesh, N_GP_T, cell_long_term, cell_NJ, cell_NNTJ, cnt_cell_list,
                                ef_init_n, ef_init_p,
                                Dirichlet_list, Dirichlet_value,
                                E_list, Ec, Eg, TOL_ef, TOL_du,
                                dos_GP_list, n_GP_list, p_GP_list, doping_GP_list, u_init, dof_amount)
    if mode == 2:
        phi = poisson_linear(info_mesh, cell_long_term, cell_NJ, Dirichlet_list, Dirichlet_value,
                       dof_amount, doping_GP_list, n_GP_list, p_GP_list)
    if mode == 3:
        poisson_analytical()
    return phi


def poisson_nonlinear(info_mesh, N_GP_T, cell_long_term, cell_NJ, cell_NNTJ, cnt_cell_list,
                      ef_init_n, ef_init_p,
                      Dirichlet_list, Dirichlet_value, E_list, Ec, Eg, TOL_ef, TOL_du,
                      dos_GP_list, n_GP_list, p_GP_list, doping_GP_list, u_init, dof_amount):
    print("non-linear poisson solver(with ef solver) start")
    # call a mapping function to map the u_init which is defined following the dof order to
    # u_init_cell which is defined following the cell structure
    u_init_cell = map_tocell(info_mesh, u_init)

    # solve the fermi energy level on gauss points of every cell
    ef_n, ef_p, ef_flag = \
        ef_solver(u_init_cell, N_GP_T, dos_GP_list, n_GP_list, p_GP_list, ef_init_n, ef_init_p,
                  cnt_cell_list, E_list, Ec, Eg, TOL_ef)
    u_k = copy.deepcopy(u_init)
    if ef_flag:
        print("ef has been solved successfully")
        # start the newton iteration now
        poisson_iter = 0
        while 1:
            poisson_iter += 1
            u_k_cell = map_tocell(info_mesh, u_k)
            A, b = assembly_nonlinear(info_mesh, N_GP_T, cell_long_term, cell_NJ, cell_NNTJ, Dirichlet_list,
                                      u_k_cell, dof_amount, ef_n, ef_p, dos_GP_list, E_list, Ec, Eg, doping_GP_list)
            du = np.linalg.solve(A.get_value(), b.get_value())
            norm_du = np.linalg.norm(du, ord=2)
            print("poisson iter:", poisson_iter, "error is:", norm_du)
            if norm_du < TOL_du:
                break
            u_k = u_k + du
        # print("poisson finished")
    else:
        u = poisson_linear(info_mesh, cell_long_term, cell_NJ, Dirichlet_list, Dirichlet_value,
                             dof_amount, doping_GP_list, n_GP_list, p_GP_list)
        u_k = u * 0.5 + u_k * 0.5
    return u_k


def poisson_linear(info_mesh, cell_long_term, cell_NJ, Dirichlet_list, Dirichlet_value,
                   dof_amount, doping_GP_list, n_GP_list, p_GP_list):
    print("linear poisson solver start")
    A, b = assembly_linear(info_mesh, cell_long_term, cell_NJ, Dirichlet_list, Dirichlet_value,
                           dof_amount, doping_GP_list, n_GP_list, p_GP_list)
    u = np.linalg.solve(A.get_value(), b.get_value())
    return u


def poisson_analytical():
    return
