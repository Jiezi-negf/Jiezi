# Copyright 2021 Jiezi authors.
#
# This file is part of Jiezi. It is subject to the license terms in the file
# LICENSE.md found in the top-level directory of this distribution. A list of
# Jiezi authors can be found in the file AUTHORS.md at the top-level directory
# of this distribution.
# ==============================================================================


from Jiezi.LA.matrix_numpy import matrix_numpy
from Jiezi.LA import operator as op


def surface_gf(E, eta, H00, H10, S00, iter_max, TOL):
    w = op.scamulmat(complex(E, eta), S00)
    iter_c = 0
    alpha = H10.dagger()
    beta = H10
    epsilon = H00
    epsilon_s = H00
    G00 = matrix_numpy()
    while iter_c < iter_max:
        alpha_new = op.trimatmul(alpha, op.inv(op.addmat(w, epsilon.nega())), \
                                 alpha, type="nnn")
        beta_new = op.trimatmul(beta, op.inv(op.addmat(w, epsilon.nega())), \
                                beta, type="nnn")
        epsilon_new = op.addmat(epsilon, \
                                op.trimatmul(alpha, op.inv(op.addmat(w, epsilon.nega())), \
                                beta, type="nnn"), \
                                op.trimatmul(beta, op.inv(op.addmat(w, epsilon.nega())), \
                                alpha, type="nnn"))
        epsilon_s_new = op.addmat(epsilon, \
                                  op.trimatmul(alpha, op.inv(op.addmat(w, epsilon.nega())), \
                                  beta, type="nnn"))

        # calculate the sum of modular squares of all elements of matrix "alpha"
        row = alpha_new.get_size()[0]
        column = alpha_new.get_size()[1]

        sum = 0.0
        for i in range(row):
            for j in range(column):
                sum += alpha_new.get_value(row, column).real**2 + \
                       alpha_new.get_value(row, column).imag**2
        if sum < TOL:
            G00 = op.inv(op.addmat(w, epsilon_s_new.nega()))
            break
        else:
            alpha.copy(alpha_new)
            beta.copy(beta_new)
            epsilon.copy(epsilon_new)
            epsilon_s.copy(epsilon_s_new)

    return G00
