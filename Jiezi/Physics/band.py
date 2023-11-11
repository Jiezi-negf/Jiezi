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
from Jiezi.Physics.hamilton import hamilton


def subband(H: hamilton, k):
    """
    this function is to calculate the eigen energy with the specific "k"
    the formula is: [H_{i-1,i}*exp(-j*ka)+H_{i,i}+H_{i,i+1}*exp(j*ka)] \psi =ES_{i,i}\psi
    :param H: hamilton object
    :param k: k point in K-space
    :return: subband[i] are eigen energy, U[i] are eigen vectors
    """
    Hii = H.get_Hii()
    Hi1 = H.get_Hi1()
    Sii = H.get_Sii()
    Si1 = H.get_Si1()
    sub_band = []
    U = []
    for i in range(len(Hii)):
        if i == 0 or i == len(Hii) - 1:
            H_temp = op.addmat(Hii[1], op.scamulmat(np.exp(-k * 1j), Hi1[1].dagger()),
                               op.scamulmat(np.exp(k * 1j), Hi1[1]))
            S_temp = op.addmat(Sii[1], op.scamulmat(np.exp(-k * 1j), Si1[1].dagger()),
                               op.scamulmat(np.exp(k * 1j), Si1[1]))
        else:
            H_temp = op.addmat(Hii[i], op.scamulmat(np.exp(-k * 1j), Hi1[i].dagger()),
                               op.scamulmat(np.exp(k * 1j), Hi1[i + 1]))
            S_temp = op.addmat(Sii[i], op.scamulmat(np.exp(-k * 1j), Si1[i].dagger()),
                               op.scamulmat(np.exp(k * 1j), Si1[i + 1]))
        H_temp = op.matmulmat(op.inv(S_temp), H_temp)
        sub_band.append(H_temp.eigenvalue())
        U.append(H_temp.eigenvec())
    return sub_band, U


def subband_defect(H: hamilton, k, cell_start, cell_repeat):
    Hii_defect, Hi1_defect, Sii_defect, Si1_defect = H.H_defect_band(cell_start, cell_repeat)
    H_temp = op.addmat(Hii_defect, op.scamulmat(np.exp(-k * 1j * cell_repeat), Hi1_defect.dagger()),
                       op.scamulmat(np.exp(k * 1j * cell_repeat), Hi1_defect))
    S_temp = op.addmat(Sii_defect, op.scamulmat(np.exp(-k * 1j * cell_repeat), Si1_defect.dagger()),
                       op.scamulmat(np.exp(k * 1j * cell_repeat), Si1_defect))
    H_temp = op.matmulmat(op.inv(S_temp), H_temp)
    sub_band = H_temp.eigenvalue()
    U = H_temp.eigenvec()
    return sub_band, U


def get_EcEg(H: hamilton):
    """
    compute bottom of conduction band energy Ec and band gap Eg
    :param H: hamilton object
    :return: Ec and Eg
    """
    Hii = H.get_Hii()
    Hi1 = H.get_Hi1()
    Sii = H.get_Sii()
    Si1 = H.get_Si1()
    H_temp = op.addmat(Hii[0], Hi1[0].dagger(), Hi1[1])
    S_temp = op.addmat(Sii[0], Si1[0].dagger(), Si1[1])
    H_temp = op.matmulmat(op.inv(S_temp), H_temp)
    energy_list = H_temp.eigenvalue().get_value()
    Ec = 0.0
    Eg = 0.0
    for i in range(len(energy_list)):
        energy = energy_list[i]
        if energy < 0:
            continue
        else:
            Ec = energy
            Eg = energy - energy_list[i-1]
            break
    return Ec, Eg


def band_structure(H: hamilton, start, end, step):
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
        sub_band, U = subband(H, k)
        band.append(sub_band[0])
    return k_total, band


def band_structure_defect(H: hamilton, start, end, step, cell_start, cell_repeat):
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
        sub_band, U = subband_defect(H, k, cell_start, cell_repeat)
        band.append(sub_band)
    return k_total, band

