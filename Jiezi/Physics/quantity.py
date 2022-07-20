# Copyright 2021 Jiezi authors.
#
# This file is part of Jiezi. It is subject to the license terms in the file
# LICENSE.md found in the top-level directory of this distribution. A list of
# Jiezi authors can be found in the file AUTHORS.md at the top-level directory
# of this distribution.
# ==============================================================================

from Jiezi.Physics.common import *
from Jiezi.LA import operator as op
import math


def quantity(E_list, G_R_fullE, G_lesser_fullE, G_greater_fullE, G1i_lesser_fullE,
             Sigma_left_lesser_fullE, Sigma_left_greater_fullE,
             Sigma_right_lesser_fullE, Sigma_right_greater_fullE,
             Hi1, volume_cell):
    n_tol = []
    p_tol = []
    J = []
    dos = []
    # number of layer
    num_layer = len(G_lesser_fullE[0])
    # compute J_L
    G_J_L = []
    for ee in range(len(E_list)):
        G_J_L.append(op.addmat(op.matmulmat(G_greater_fullE[ee][0],
                                            Sigma_left_lesser_fullE[ee]),
                               op.matmulmat(G_lesser_fullE[ee][0],
                                            Sigma_left_greater_fullE[ee]).nega()).tre())
    J_L = q_unit * integral(E_list, G_J_L) / math.pi / h_bar
    J.append(J_L.real)

    # compute n, p, and J(i->i+1)
    for i in range(num_layer):
        # G_n_i[ee]: energy is ee, layer is i. i is constant, ee is different
        G_n_i = []
        G_p_i = []
        # # number of sites per layer
        # num_site = G_lesser_fullE[0][i].get_size()[0]

        # compute the function G(ee) in location i, which will be integrated
        for ee in range(len(E_list)):
            G_n_i.append(G_lesser_fullE[ee][i].tre() / volume_cell)
            G_p_i.append(G_greater_fullE[ee][i].tre() / volume_cell)
        # n_i, p_i: n of location/layer i
        n_i = integral(E_list, G_n_i) / math.pi
        p_i = integral(E_list, G_p_i) / math.pi
        n_tol.append(n_i)
        p_tol.append(p_i)

    for i in range(0, num_layer - 1):
        # compute J(i->i+1)
        G_J_i = []
        # compute the function G_J_i(ee) in location i, which will be integrated
        for ee in range(len(E_list)):
            G_J_i.append(-2.0 * op.matmulmat(Hi1[i+1], G1i_lesser_fullE[ee][i]).imaginary().tre())

        J_i = q_unit * integral(E_list, G_J_i) / math.pi / h_bar
        J.append(J_i)

    # compute J_R
    G_J_R = []
    for ee in range(len(E_list)):
        G_J_R.append(op.addmat(op.matmulmat(G_greater_fullE[ee][num_layer - 1],
                                            Sigma_right_lesser_fullE[ee]),
                               op.matmulmat(G_lesser_fullE[ee][num_layer - 1],
                                            Sigma_right_greater_fullE[ee]).nega()).tre())
    J_R = - q_unit * integral(E_list, G_J_R) / math.pi / h_bar
    J.append(J_R.real)

    # compute density of states(DOS)
    for ee in range(len(E_list)):
        dos_ee = []
        for zz in range(num_layer):
            dos_ee.append(- G_R_fullE[ee][zz].tre().imag / volume_cell)
        dos.append(dos_ee)

    return n_tol, p_tol, J, dos