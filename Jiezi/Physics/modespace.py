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
    """
    transform the real-space matrix to mode-space matrix by U, which is the eigenvector of H(k=0)
    :param H: real-space hamiltonian matrix
    :param U: matrix composed of eigenvector at k=0
    :param nm: the amount of bands which should be remained in the transform process, the choice of bands
                is around the fermi energy EF.
    :return: new Hii Hi1 Sii(the data structure is the same as old H, but the dimension is reduced)
            and form_factor which will be used in phonon model
    """
    Hii = H.get_Hii().copy()
    Hi1 = H.get_Hi1().copy()
    Sii = H.get_Sii().copy()
    nz = len(Hii)
    U_new = []
    U_new_i = matrix_numpy()
    U_unitary = []
    # use qr decomposition to do gram-schmidt process for original U which is not unitary now
    for i in range(nz):
        U_unitary_i = op.qr_decomp(U[i])
        U_unitary.append(U_unitary_i)
        nn = U[i].get_size()[0]
        U_new_i.copy(U_unitary_i.get_value(0, nn, nn//2 - nm//2, nn//2 + nm//2))
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
    return Hii, Hi1, Sii, form_factor, U_new