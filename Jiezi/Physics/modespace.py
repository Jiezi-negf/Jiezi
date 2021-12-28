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


def mode_space(H: hamilton, U, part: str, nm: int):
    Hii = H.get_Hii()
    Hi1 = H.get_Hi1()
    Sii = H.get_Sii()
    nz = len(Hii)
    U_new = []
    # TODO: Hii_new, Hi1_new and Sii_new seem to be unnecessary, one can modify directly
    # on Hii, Hi1 and Sii instead
    Hii_new = []
    Hi1_new = []
    Sii_new = []
    # TODO: think initialize matrix from base class
    U_new_i = matrix_numpy()
    for i in range(nz):
        nn = U[i].get_size()[0]
        if part == "low":
            # remain the left nm columns of matrix(lower energy)
            U_new_i.copy(U[i].get_value(0, nn - 1, 0, nm - 1))
        else:
            # remain the right nm column of matrix(higher energy)
            U_new_i.copy(U[i].get_value(0, nn - 1, nn - nm - 1, nn))
        U_new.append(U_new_i)
    # transfer the Hii and Sii
    for i in range(nz):
        Hii_new_i = op.trimatmul(U_new[i], Hii[i], U_new[i], type="cnn")
        Sii_new_i = op.trimatmul(U_new[i], Sii[i], U_new[i], type="cnn")
        Hii_new.append(Hii_new_i)
        Sii_new.append(Sii_new_i)
    # the head of Hi1 follow the different rule as follows
    Hi1_new_i = op.trimatmul(U_new[0], Hi1[0], U_new[0], type="cnn")
    Hi1_new.append(Hi1_new_i)
    # transfer the Hi1
    for i in range(1, nz):
        Hi1_new_i = op.trimatmul(U_new[i - 1], Hi1[i], U_new[i], type="cnn")
        Hi1_new.append(Hi1_new_i)
    # the tail of Hi1 follow the different rule as follows
    Hi1_new_i = op.trimatmul(U_new[nz - 1], Hi1[nz], U_new[nz - 1], type="cnn")
    Hi1_new.append(Hi1_new_i)
    # compute form_factor
    form_factor = []
    for i in range(nz):
        form_factor_i = op.matmul_sym(U_new[i].conjugate(), U_new[i])
        form_factor.append(form_factor_i)
    return Hii_new, Hi1_new, Sii_new, form_factor
