# Copyright 2021 Jiezi authors.
#
# This file is part of Jiezi. It is subject to the license terms in the file
# LICENSE.md found in the top-level directory of this distribution. A list of
# Jiezi authors can be found in the file AUTHORS.md at the top-level directory
# of this distribution.
# ==============================================================================

import os
import sys
sys.path.append(os.path.abspath(__file__ + "/../../.."))
import math
from Jiezi.LA.matrix_numpy import matrix_numpy
from Jiezi.LA.vector_numpy import vector_numpy
from Jiezi.LA import operator as op
import numpy as np
from Jiezi.Physics import hamilton, band
from Jiezi.Physics import w90_trans as w90
import matplotlib.pyplot as plt

path_Files = "/home/zjy/Jiezi/Jiezi/Files/"
path_hr = path_Files + "wannier90_hr_new.dat"
r_set, hr_set = w90.read_hamiltonian(path_hr)

# extract hamiltonian matrix of CNT
hr_cnt_set = [None] * len(hr_set)
hr_graphene_set = [None] * len(hr_set)
hr_total_set = [None] * len(hr_set)
for i in range(len(hr_set)):
    hr_cnt_set[i] = matrix_numpy()
    hr_graphene_set[i] = matrix_numpy()
    hr_total_set[i] = matrix_numpy()
    hr_temp = matrix_numpy()
    hr_temp.copy(hr_set[i].get_value())
    # swap for new version
    swap_list = [(3, 55), (13, 56), (15, 41), (35, 43)]
    for j in range(len(swap_list)):
        hr_temp = hr_temp.swap_index(swap_list[j][0], swap_list[j][1])
    ## swapped matrix
    hr_cnt_set[i].copy(hr_temp.get_value(40, 72, 40, 72))
    hr_graphene_set[i].copy(hr_temp.get_value(0, 40, 0, 40))
    hr_total_set[i].copy(hr_temp.get_value(0, 72, 0, 72))
r_1, r_2, r_3, k_1, k_2, k_3 = w90.latticeVectorInit()
k_coord = [0.33333, 0, 0]
num_supercell = 2
length_single_cell = 4.27
num_cell = 60
L = length_single_cell * num_supercell

Mii_total, Mi1_total = w90.w90_supercell_matrix(r_set, hr_total_set, r_1, r_2, r_3, k_1, k_2, k_3,
                                            k_coord, num_supercell)
Mii_cnt, Mi1_cnt = w90.w90_supercell_matrix(r_set, hr_cnt_set, r_1, r_2, r_3, k_1, k_2, k_3,
                                            k_coord, num_supercell)

Ec, Ev, Eg = band.get_EcEg(Mii_cnt, Mi1_cnt)
Emid = (Ec + Ev) / 2


# shift the whole band to the middle of energy axis
energy_shift = 0 - Emid
ele_total = matrix_numpy(Mii_total.get_size()[0], Mii_total.get_size()[1])
ele_total.identity()
ele_cnt = matrix_numpy(Mii_cnt.get_size()[0], Mii_cnt.get_size()[1])
ele_cnt.identity()
Mii_total = op.addmat(Mii_total, op.scamulmat(energy_shift, ele_total))
Mii_cnt = op.addmat(Mii_cnt, op.scamulmat(energy_shift, ele_cnt))


# k_points = np.arange(-np.pi / L, np.pi / L, 2 * np.pi / (num_cell / num_supercell) / L / 1000)
k_points = np.arange(-np.pi / L, np.pi / L, 2 * np.pi / (num_cell / num_supercell) / L / 20)
bandArray = hamilton.compute_band(Mii_cnt, Mi1_cnt, L, k_points)


# plot band
num_band = bandArray.shape[1]
for i in range(num_band):
    y = bandArray[:, i:i+1]
    plt.plot(k_points, y)
    ax = plt.gca()
    ax.set_aspect(0.1)
plt.show()


# start = -5
# end = 5
# step = 0.001
# E_list = np.arange(start, end, step)
# dos0 = band.band2dos(bandArray, E_list)
# dos = list(dos0[:, 1] * 2 / 1000 / (num_cell / num_supercell))
# plt.plot(dos0[:, 0], dos)
# plt.show()