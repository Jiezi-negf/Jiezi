# Copyright 2021 Jiezi authors.
#
# This file is part of Jiezi. It is subject to the license terms in the file
# LICENSE.md found in the top-level directory of this distribution. A list of
# Jiezi authors can be found in the file AUTHORS.md at the top-level directory
# of this distribution.
# ==============================================================================

import numpy as np
import copy
from Jiezi.FEM.map import map_tocell
from Jiezi.FEM.ef_solver import ef_solver, ef_solver_analytical, ef_solver_degenerate
from Jiezi.FEM.assembly import (assembly_nonlinear, assembly_linear, assembly_analytical, assembly_init,
                                assembly_degenerate)
from Jiezi.Physics.common import time_it
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix
import time

@ time_it
def poisson(mode, info_mesh, N_GP_T, cell_long_term, cell_NJ, cell_NNTJ, cnt_cell_list,
            ef_init_n, ef_init_p, mark_list,
            Dirichlet_list, Dirichlet_value, E_list, Ec, Eg, TOL_ef, TOL_du, iter_NonLinearPoisson_max,
            dos_GP_list, n_GP_list, p_GP_list, doping_GP_list, fixedCharge_GP_list, u_init, dof_amount, Nc=0, Nv=0):
    print("poisson solver start")
    phi = None
    if mode == 1:
        phi = poisson_nonlinear(info_mesh, N_GP_T, cell_long_term, cell_NJ, cell_NNTJ, cnt_cell_list,
                                ef_init_n, ef_init_p, mark_list,
                                Dirichlet_list, Dirichlet_value,
                                E_list, Ec, Eg, TOL_ef, TOL_du, iter_NonLinearPoisson_max,
                                dos_GP_list, n_GP_list, p_GP_list, doping_GP_list, fixedCharge_GP_list,
                                u_init, dof_amount)
    if mode == 2:
        phi = poisson_linear(info_mesh, cell_long_term, cell_NJ, Dirichlet_list, Dirichlet_value,
                       dof_amount, doping_GP_list, fixedCharge_GP_list, n_GP_list, p_GP_list)
    if mode == 3:
        Ev = Ec - Eg
        phi = poisson_analytical(info_mesh, N_GP_T, cell_long_term, cell_NJ, cell_NNTJ, cnt_cell_list,
                      ef_init_n, ef_init_p, mark_list,
                      Dirichlet_list, Dirichlet_value, Ec, Ev, TOL_du, iter_NonLinearPoisson_max,
                      n_GP_list, p_GP_list, doping_GP_list, fixedCharge_GP_list, u_init, dof_amount, Nc, Nv)
    if mode == 4:
        Ev = Ec - Eg
        phi = poisson_degenerate(info_mesh, N_GP_T, cell_long_term, cell_NJ, cell_NNTJ, cnt_cell_list,
                                 ef_init_n, ef_init_p, mark_list,
                                 Dirichlet_list, Dirichlet_value, Ec, Ev, TOL_du, iter_NonLinearPoisson_max,
                                 n_GP_list, p_GP_list, doping_GP_list, fixedCharge_GP_list, u_init, dof_amount, Nc, Nv)
    return phi


def poisson_nonlinear(info_mesh, N_GP_T, cell_long_term, cell_NJ, cell_NNTJ, cnt_cell_list,
                      ef_init_n, ef_init_p, mark_list,
                      Dirichlet_list, Dirichlet_value, E_list, Ec, Eg, TOL_ef, TOL_du, iter_NonLinearPoisson_max,
                      dos_GP_list, n_GP_list, p_GP_list, doping_GP_list, fixedCharge_GP_list, u_init, dof_amount):

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
        while poisson_iter < iter_NonLinearPoisson_max:
            poisson_iter += 1
            u_k_cell = map_tocell(info_mesh, u_k)
            A, b = assembly_nonlinear(info_mesh, N_GP_T, cell_long_term, cell_NJ, cell_NNTJ, Dirichlet_list, mark_list,
                                      u_k_cell, dof_amount, ef_n, ef_p, dos_GP_list, E_list, Ec, Eg, doping_GP_list,
                                      fixedCharge_GP_list)
            du = sparseSolve(A, b.get_value())
            norm_du = np.linalg.norm(du, ord=2)
            print("poisson iter:", poisson_iter, "error is:", norm_du)
            if norm_du < TOL_du:
                break
            u_k = u_k + du

        # if non-linear Poisson can't converge, then start linear poisson solver
        if poisson_iter == iter_NonLinearPoisson_max:
            print("non-linear poisson solver reached the iteration times limit!")
            u_k = poisson_linear(info_mesh, cell_long_term, cell_NJ, Dirichlet_list, Dirichlet_value,
                                 dof_amount, doping_GP_list, fixedCharge_GP_list, n_GP_list, p_GP_list)
    else:
        u_k = poisson_linear(info_mesh, cell_long_term, cell_NJ, Dirichlet_list, Dirichlet_value,
                             dof_amount, doping_GP_list, fixedCharge_GP_list, n_GP_list, p_GP_list)
    return u_k


def poisson_linear(info_mesh, cell_long_term, cell_NJ, Dirichlet_list, Dirichlet_value,
                   dof_amount, doping_GP_list, fixedCharge_GP_list, n_GP_list, p_GP_list):
    print("linear poisson solver start")
    A, b = assembly_linear(info_mesh, cell_long_term, cell_NJ, Dirichlet_list, Dirichlet_value,
                           dof_amount, doping_GP_list, fixedCharge_GP_list, n_GP_list, p_GP_list)
    u = np.linalg.solve(A.get_value(), b.get_value())
    return u


def poisson_analytical(info_mesh, N_GP_T, cell_long_term, cell_NJ, cell_NNTJ, cnt_cell_list,
                      ef_init_n, ef_init_p, mark_list,
                      Dirichlet_list, Dirichlet_value, Ec, Ev, TOL_du, iter_NonLinearPoisson_max,
                      n_GP_list, p_GP_list, doping_GP_list, fixedCharge_GP_list, u_init, dof_amount, Nc, Nv):

    print("analytical poisson solver(with ef solver analytical) start")
    # call a mapping function to map the u_init which is defined following the dof order to
    # u_init_cell which is defined following the cell structure
    u_init_cell = map_tocell(info_mesh, u_init)
    # solve the fermi energy level on gauss points of every cell
    ef_n, ef_p = \
        ef_solver_analytical(u_init_cell, N_GP_T, Nc, Nv, n_GP_list, p_GP_list, cnt_cell_list, Ec, Ev,
                             ef_init_n, ef_init_p)
    u_k = copy.deepcopy(u_init)
    print("ef has been solved successfully")
    # start the newton iteration now
    poisson_iter = 0
    while poisson_iter < iter_NonLinearPoisson_max:
        poisson_iter += 1
        u_k_cell = map_tocell(info_mesh, u_k)
        A, b = assembly_analytical(info_mesh, N_GP_T, cell_long_term, cell_NJ, cell_NNTJ, Dirichlet_list, mark_list,
                                  u_k_cell, dof_amount, ef_n, ef_p, Ec, Ev, doping_GP_list,
                                  fixedCharge_GP_list, Nc, Nv)
        du = sparseSolve(A.get_value(), b.get_value())
        norm_du = np.linalg.norm(du, ord=2)
        print("poisson iter:", poisson_iter, "error is:", norm_du)
        if norm_du < TOL_du:
            break
        u_k = u_k + du
        # if non-linear Poisson can't converge, then start linear poisson solver
        if poisson_iter == iter_NonLinearPoisson_max:
            print("non-linear poisson solver reached the iteration times limit!")
            u_k = poisson_linear(info_mesh, cell_long_term, cell_NJ, Dirichlet_list, Dirichlet_value,
                                 dof_amount, doping_GP_list, fixedCharge_GP_list, n_GP_list, p_GP_list)
    return u_k

def poisson_degenerate(info_mesh, N_GP_T, cell_long_term, cell_NJ, cell_NNTJ, cnt_cell_list,
                      ef_init_n, ef_init_p, mark_list,
                      Dirichlet_list, Dirichlet_value, Ec, Ev, TOL_du, iter_NonLinearPoisson_max,
                      n_GP_list, p_GP_list, doping_GP_list, fixedCharge_GP_list, u_init, dof_amount, Nc, Nv):

    print("degenerate poisson solver(with ef solver degenerate) start")
    # call a mapping function to map the u_init which is defined following the dof order to
    # u_init_cell which is defined following the cell structure
    u_init_cell = map_tocell(info_mesh, u_init)
    # solve the fermi energy level on gauss points of every cell
    ef_n, ef_p = \
        ef_solver_degenerate(u_init_cell, N_GP_T, Nc, Nv, n_GP_list, p_GP_list, cnt_cell_list, Ec, Ev,
                             ef_init_n, ef_init_p)
    u_k = copy.deepcopy(u_init)
    print("ef has been solved successfully")
    # start the newton iteration now
    poisson_iter = 0
    while poisson_iter < iter_NonLinearPoisson_max:
        poisson_iter += 1
        u_k_cell = map_tocell(info_mesh, u_k)
        A, b = assembly_degenerate(info_mesh, N_GP_T, cell_long_term, cell_NJ, cell_NNTJ, Dirichlet_list, mark_list,
                                  u_k_cell, dof_amount, ef_n, ef_p, Ec, Ev, doping_GP_list,
                                  fixedCharge_GP_list, Nc, Nv)
        du = sparseSolve(A.get_value(), b.get_value())
        norm_du = np.linalg.norm(du, ord=2)
        print("poisson iter:", poisson_iter, "error is:", norm_du)
        if norm_du < TOL_du:
            break
        u_k = u_k + du
        # if non-linear Poisson can't converge, then start linear poisson solver
        if poisson_iter == iter_NonLinearPoisson_max:
            print("non-linear poisson solver reached the iteration times limit!")
            u_k = poisson_linear(info_mesh, cell_long_term, cell_NJ, Dirichlet_list, Dirichlet_value,
                                 dof_amount, doping_GP_list, fixedCharge_GP_list, n_GP_list, p_GP_list)
    return u_k

def poissonGuess(modeGuess, info_mesh, cell_long_term, cell_NJ, Dirichlet_list, Dirichlet_value, dof_amount,
                 doping_GP_list, fixedCharge_GP_list, geo_para, phi_l, phi_r, V_gate):
    if modeGuess == 1:
        print("the guess of phi is constant except boundary")
        phi_guess = np.ones([dof_amount, 1]) * 0.7
        for type_i in range(len(Dirichlet_list)):
            for i in range(len(Dirichlet_list[type_i])):
                phi_guess[Dirichlet_list[type_i][i], 0] = Dirichlet_value[type_i]
        res = phi_guess
    elif modeGuess == 2:
        print("the guess of phi is from laplace equation solver")
        A, b = assembly_init(info_mesh, cell_long_term, cell_NJ, Dirichlet_list, Dirichlet_value, dof_amount,
                             doping_GP_list, fixedCharge_GP_list)
        u = np.linalg.solve(A.get_value(), b.get_value())
        res = u
    else:
        print("the guess of phi is for physical doping")
        zlength_oxide = geo_para[4]
        z_translation = geo_para[5]
        z_isolation = geo_para[6]
        u_vec = np.zeros((dof_amount, 1))
        cell_amount = len(info_mesh)
        for i in range(cell_amount):
            cell_dof_index = list(info_mesh[i].keys())
            for j in range(4):
                dof_index = cell_dof_index[j]
                x, y, z = info_mesh[i][dof_index]
                u_vec[dof_index, 0] = guessPhiforDope(z, zlength_oxide, z_translation, z_isolation,
                                                      phi_l, phi_r, V_gate)
        res = u_vec
    return res

def guessPhiforDope(z, zlength_oxide, z_translation, z_isolation, phi_l, phi_r, V_gate):
    res = 0
    z_1 = z_translation - z_isolation
    z_2 = z_translation
    z_3 = z_2 + zlength_oxide
    z_4 = z_3 + z_isolation
    if z <= z_1:
        res = phi_l
    elif z_1 < z < z_2:
        res = (V_gate - phi_l) / z_isolation * z + \
              (phi_l * z_2 - V_gate * z_1) / z_isolation
    elif z_2 <= z < z_3:
        res = V_gate
    elif z_3 <= z < z_4:
        res = (phi_r - V_gate) / z_isolation * z + \
              (V_gate * z_4 - phi_r * z_3) / z_isolation
    else:
        res = phi_r
    return res

def sparseSolve(matrixA, matrixb):
    time1 = time.time()
    A_sparse = matrixA.tocsr()
    time2 = time.time()
    print("csr trans time:", time2 - time1)
    print("non zero elements amount:", A_sparse.count_nonzero())
    time3 = time.time()
    du = spsolve(A_sparse, matrixb).reshape((matrixb.shape[0], 1))
    time4 = time.time()
    print("spsolve time:", time4 - time3)
    return du
#def guessPhiforDope(z, zlength_oxide, z_translation, z_isolation, phi_l, phi_r, V_gate):
#    res = 0
#    z_1 = z_translation - z_isolation
#    z_2 = z_translation
#    z_3 = z_2 + zlength_oxide
#    z_4 = z_3 + z_isolation
#    z_mid_l = (z_1 + z_2) / 2
#    z_mid_r = (z_3 + z_4) / 2
#    phi_mid_l =  (V_gate + phi_l) / 2
#    phi_mid_r = (V_gate + phi_r) / 2
#   x1l, x2l = guess_exp2(z_1, z_mid_l, phi_l, phi_mid_l, 'l')
#    x1r, x2r = guess_exp2(z_mid_r, z_4, phi_mid_r, phi_r, 'r')
#    if z <= z_1:
#        res = phi_l
#    elif z_1 < z <= z_mid_l:
#        res = x1l - x2l * np.exp(z)
#    elif z_mid_l < z <= z_2:
#        res = (V_gate - phi_mid_l) / (z_2 - z_mid_l) * z + \
#               (phi_mid_l * z_2 - V_gate * z_mid_l) / (z_2 - z_mid_l)
#    elif z_2 <z <= z_3:
#        res = V_gate
#    elif z_3 < z <= z_mid_r:
#       res = (phi_mid_r - V_gate) / (z_mid_r - z_3) * z + \
#               (V_gate * z_mid_r - phi_mid_r * z_3) / (z_mid_r - z_3)
#    elif z_mid_r < z <= z_4:
#        res = x1r - x2r * np.exp(-z)
#    else:
#        res = phi_r
#    return res


#def guess_exp2(a1, a2, b1, b2, flag):
#    if flag == 'l':
#        x2 = (b2 - b1) / (np.exp(a1) - np.exp(a2))
#       x1 = b1 + x2 * np.exp(a1)
#    else:
#        x2 = (b2 - b1) / (np.exp(-a1) - np.exp(-a2))
#        x1 = b1 + x2 * np.exp(-a1)
#   return x1, x2



