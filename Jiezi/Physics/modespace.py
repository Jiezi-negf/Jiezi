# Copyright 2021 Jiezi authors.
#
# This file is part of Jiezi. It is subject to the license terms in the file
# LICENSE.md found in the top-level directory of this distribution. A list of
# Jiezi authors can be found in the file AUTHORS.md at the top-level directory
# of this distribution.
# ==============================================================================
from Jiezi.Physics.hamilton import hamilton
from Jiezi.LA.matrix_numpy import matrix_numpy
from Jiezi.LA import operator as op


def mode_space(H: hamilton, U, nm: int):
    Hii = H.get_Hii()
    Hi1 = H.get_Hi1()
    Sii = H.get_Sii()
    nz = len(Hii)
    U_new = []
    U_new_i = matrix_numpy()
    for i in range(nz):
        nn = U[i].get_size()[0]
        U_new_i.copy(U[i].get_value(0, nn, nn//2 - nm//2, nn//2 + nm//2))
        U_new.append(U_new_i)
    # transfer the Hii and Sii
    for i in range(nz):
        Hii[i] = op.trimatmul(U_new[i], Hii[i], U_new[i], type="cnn")
        Sii[i] = op.trimatmul(U_new[i], Sii[i], U_new[i], type="cnn")

    # the head of Hi1 follow the different rule as follows
    Hi1[0] = op.trimatmul(U_new[0], Hi1[0], U_new[0], type="cnn")

    # transfer the Hi1
    for i in range(1, nz):
        Hi1[i] = op.trimatmul(U_new[i - 1], Hi1[i], U_new[i], type="cnn")

    # the tail of Hi1 follow the different rule as follows
    Hi1[nz] = op.trimatmul(U_new[nz - 1], Hi1[nz], U_new[nz - 1], type="cnn")

    # compute form_factor
    form_factor = []
    for i in range(nz):
        form_factor_i = op.matmul_sym(U_new[i].dagger(), U_new[i])
        form_factor.append(form_factor_i)
    return Hii, Hi1, Sii, form_factor