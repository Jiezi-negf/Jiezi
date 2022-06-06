# Copyright 2021 Jiezi authors.
#
# This file is part of Jiezi. It is subject to the license terms in the file
# LICENSE.md found in the top-level directory of this distribution. A list of
# Jiezi authors can be found in the file AUTHORS.md at the top-level directory
# of this distribution.
# ==============================================================================
import math
from Jiezi.LA import operator as op
from Jiezi.LA.matrix_numpy import matrix_numpy
import numpy as np

# physical parameters
h = 4.1357e-15
h_bar = h/(2 * math.pi)
q_unit = 1e-19
mul = -0.2
mur = -2.0
# mul = -1.0
# mur = -2.0
KT = 0.026


# geometric parameters
r_inter, r_oxide, cnt_radius, width_cnt, width_oxide, z_total, zlength_oxide = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
# material parameters
epsilon_0 = 1.0
epsilon_air = 2.0 * epsilon_0
epsilon_cnt = 3.0 * epsilon_0
epsilon_oxide = 4.0 * epsilon_0


def bose(E, BOSE=1.0, TEMP=1.0):
    return 1.0 / (math.exp(E /KT) - 1.0)


def fermi(x):
    return 1.0 / (1.0 + math.exp(x / KT))


def heaviside(x):
    if x >= 0.0:
        return 1.0
    else:
        return 0.0


def integral(E, G):
    E_num = len(E)
    result = 0.0
    for ee in range(E_num - 1):
        result += (G[ee] + G[ee + 1]) * (E[ee + 1] - E[ee]) / 2.0
    return result


def ifdagger(mat: matrix_numpy):
    row, col = mat.get_size()
    delta = op.addmat(mat, mat.dagger().nega())
    error = 0.0
    for i in range(row):
        for j in range(col):
            error += np.sqrt(delta.get_value(i, j).imag ** 2 + delta.get_value(i, j).real ** 2)
    return error