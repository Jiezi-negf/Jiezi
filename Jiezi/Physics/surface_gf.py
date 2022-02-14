# Copyright 2021 Jiezi authors.
#
# This file is part of Jiezi. It is subject to the license terms in the file
# LICENSE.md found in the top-level directory of this distribution. A list of
# Jiezi authors can be found in the file AUTHORS.md at the top-level directory
# of this distribution.
# ==============================================================================


from Jiezi.LA.matrix_numpy import matrix_numpy
from Jiezi.LA import operator as op
import numpy as np


def surface_gf(E, eta, H00, H10, S00, iter_max=50, TOL=1e-10):
    print("start the surface GF loop")
    w = op.scamulmat(complex(E, eta), S00)
    iter_c = 0
    alpha = matrix_numpy()
    alpha.copy(H10.get_value())
    beta = H10.dagger()
    epsilon = matrix_numpy()
    epsilon.copy(H00.get_value())
    epsilon_s = matrix_numpy()
    epsilon_s.copy(H00.get_value())
    G00 = matrix_numpy()
    GBB = matrix_numpy()
    while iter_c < iter_max:
        iter_c += 1
        alpha_new = op.trimatmul(alpha,
                                 op.inv(op.addmat(w, epsilon.nega())),
                                 alpha, type="nnn")
        beta_new = op.trimatmul(beta,
                                op.inv(op.addmat(w, epsilon.nega())),
                                beta, type="nnn")
        epsilon_new = op.addmat(epsilon,
                                op.trimatmul(alpha, op.inv(op.addmat(w, epsilon.nega())),
                                             beta, type="nnn"),
                                op.trimatmul(beta, op.inv(op.addmat(w, epsilon.nega())),
                                             alpha, type="nnn"))
        epsilon_s_new = op.addmat(epsilon_s,
                                  op.trimatmul(alpha, op.inv(op.addmat(w, epsilon.nega())),
                                               beta, type="nnn"))
        # calculate the sum of modular squares of all elements of matrix "alpha"
        row = alpha_new.get_size()[0]
        column = alpha_new.get_size()[1]

        sum = 0.0
        for i in range(row):
            for j in range(column):
                sum += np.sqrt(alpha_new.get_value(i, j).real ** 2 +
                               alpha_new.get_value(i, j).imag ** 2)
        print(
            'iter number of surface GF loop is: {iter}, error of {iter} loop is: {error}'.format(iter=iter_c,
                                                                                                 error=sum))
        if sum < TOL:
            G00 = op.inv(op.addmat(w, epsilon_s_new.nega()))
            GBB = op.inv(op.addmat(w, epsilon_new.nega()))
            break
        else:
            alpha.copy(alpha_new.get_value())
            beta.copy(beta_new.get_value())
            epsilon.copy(epsilon_new.get_value())
            epsilon_s.copy(epsilon_s_new.get_value())
    return G00, GBB


def surface_gf_dumb(E, eta, H00, H10, S00, iter_max=50, TOL=1e-10):
    """
    when computing left contact, H10 is Hi1.dagger.
    H10 is Hi1 for right contact.
    Generally, eta is from 1e-6 to 2e-2, unit is eV (electron volt)
    """
    print("start the surface_gf_dumb loop")
    w = op.scamulmat(complex(E, eta), S00)
    iter_c = 0
    g_0 = op.inv(op.addmat(w, H00.nega()))
    row = g_0.get_size()[0]
    column = g_0.get_size()[1]
    G00 = matrix_numpy()
    while iter_c < iter_max:
        iter_c += 1
        g_i = op.inv(op.addmat(w, H00.nega(), op.trimatmul(H10, g_0, H10, type="nnc").nega()))
        delta = op.addmat(g_i, g_0.nega())
        sum = 0.0
        for i in range(row):
            for j in range(column):
                sum += np.sqrt(delta.get_value(i, j).real ** 2 + delta.get_value(i, j).imag ** 2)
        print(
            'iter number of surface GF loop is: {iter}, error of {iter} loop is: {error}'.format(
                iter=iter_c, error=sum))
        if sum < TOL:
            G00.copy(g_i.get_value())
            break
        else:
            g_0.copy(g_i.get_value())
    return G00

