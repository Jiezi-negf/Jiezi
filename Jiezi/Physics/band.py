# Copyright 2021 Jiezi authors.
#
# This file is part of Jiezi. It is subject to the license terms in the file
# LICENSE.md found in the top-level directory of this distribution. A list of
# Jiezi authors can be found in the file AUTHORS.md at the top-level directory
# of this distribution.
# ==============================================================================

import numpy as np
from Jiezi.LA.vector_numpy import vector_numpy
from Jiezi.LA.matrix_numpy import matrix_numpy
from Jiezi.LA import operator as op


def subband(Hii, Hi1, Sii, Si1, k):
    """
    this function is to calculate the eigen energy with the specific "k"
    the formula is: [H_{i-1,i}*exp(-j*ka)+H_{i,i}+H_{i,i+1}*exp(j*ka)] \psi =ES_{i,i}\psi
    :param Hii: Hii[i]=H_{i,i}
    :param Hi1: Hi1[i]=H_{i-1,i}
    :param Sii: base overlap
    :param k:
    :return: subband[i] are eigen energy, U[i] are eigen vectors
    """
    sub_band = []
    U = []
    for i in range(len(Hii)):
        H_temp = op.addmat(Hii[i], op.scamulmat(np.exp(-k * 1j), Hi1[i].trans()),
                           op.scamulmat(np.exp(k * 1j), Hi1[i + 1]))
        S_temp = op.addmat(Sii[i], op.scamulmat(np.exp(-k * 1j), Si1[i].trans()),
                           op.scamulmat(np.exp(k * 1j), Si1[i + 1]))
        H_temp = op.matmulmat(op.inv(S_temp), H_temp)
        sub_band.append(H_temp.eigenvalue())
        U.append(H_temp.eigenvec())
    return sub_band, U


def band_structure(Hii, Hi1, Sii, Si1, start, end, step):
    """
    plot the band structure by scanning the k from start to end
    :param start: the beginning of k
    :param end: the end of k
    :param step: the step of k
    :return: the list of k, every k has a list of subband value
    """
    k_total = np.arange(start, end, step)
    band = []
    for k in k_total:
        sub_band, U = subband(Hii, Hi1, Sii, Si1, k)
        band.append(sub_band[0])
    return k_total, band


