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
import time

# physical parameters
h = 4.1357e-15
h_bar = h/(2 * math.pi)
q_unit = 1.6e-19
mul = 0.0
mur = 0.0
# mul = -1.0
# mur = -2.0
KT = 0.026


# geometric parameters
r_inter, r_oxide, cnt_radius, width_cnt, width_oxide, z_total, zlength_oxide = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
# material parameters
epsilon_0 = 8.854e-22 / q_unit
epsilon_air = 1.0 * epsilon_0
epsilon_cnt = 6.9 * epsilon_0
epsilon_oxide = 3.9 * epsilon_0
epsilon_air_outer = 1000.0 * epsilon_0
# epsilon_air = 1.0 * epsilon_0
# epsilon_cnt = 1.0 * epsilon_0
# epsilon_oxide = 1.0 * epsilon_0
# epsilon_air_outer = 1.0 * epsilon_0

def bose(E, BOSE=1.0, TEMP=1.0):
    return 1.0 / (math.exp(E /KT) - 1.0)


def fermi(x):
    if x / KT < -709:
        res = 1.0
    elif x / KT > 709:
        res = 0.0
    else:
        res = 1.0 / (1.0 + np.exp(x / KT))
    return res


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


def time_it(func):
    def inner(*args, **kw):
        start = time.time()
        back = func(*args, **kw)
        end = time.time()
        print("Time cost of", func.__name__, "function:", end-start, "seconds")
        return back
    return inner
