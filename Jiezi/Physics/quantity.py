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


def quantity(G_lesser_fullE, G_greater_fullE, G1i_lesser_fullE, 
             Sigma_left_lesser_fullE, Sigma_left_greater_fullE, Hi1):
    n_tol = []
    p_tol = []
    J = []
    # number of layer
    num_layer = len(G_lesser_fullE[0])
    # compute J_L
    G_J_L = []
    for ee in range(len(E_list)):
        G_J_L.append(op.addmat(op.matmulmat(G_greater_fullE[ee][0], \
                                            Sigma_left_lesser_fullE[ee]),
                               op.matmulmat(G_lesser_fullE[ee][0], \
                                            Sigma_left_greater_fullE[ee]).nega()).tre())

    J_L = integral(E_list, G_J_L) / math.pi / h_bar * q_unit
    J.append(J_L)

    # compute n, p, and J(i->i+1)
    for i in range(num_layer):
        # G_n_i[ee]: energy is ee, layer is i. i is constant, ee is different
        G_n_i = []
        G_p_i = []
        # number of sites per layer
        num_site = G_lesser_fullE[0][i].get_size()[0]
        # compute the function G(ee) in location i, which will be integrated
        for ee in range(len(E_list)):
            G_n_i.append(G_lesser_fullE[ee][i].tre() / num_site)
            G_p_i.append(G_greater_fullE[ee][i].tre() / num_site)
        # n_i, p_i: n of location/layer i
        n_i = integral(E_list, G_n_i) / math.pi
        p_i = integral(E_list, G_p_i) / math.pi
        n_tol.append(n_i)
        p_tol.append(p_i)

        # compute J(i->i+1)
        G_J_i = []
        # compute the function G_J_i(ee) in location i, which will be integrated
        for ee in range(len(E_list)):
            G_J_i.append(-2 * op.matmulmat(Hi1[i+1], G1i_lesser_fullE[ee][i]).imaginary().tre())

        J_i = integral(E_list, G_J_i)/ math.pi / h_bar * q_unit
        J.append(J_i)

    return n_tol, p_tol, J
