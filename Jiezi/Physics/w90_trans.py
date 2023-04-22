# Copyright 2021 Jiezi authors.
#
# This file is part of Jiezi. It is subject to the license terms in the file
# LICENSE.md found in the top-level directory of this distribution. A list of
# Jiezi authors can be found in the file AUTHORS.md at the top-level directory
# of this distribution.
# ==============================================================================
import math
from Jiezi.LA.matrix_numpy import matrix_numpy
from Jiezi.LA.vector_numpy import vector_numpy
from Jiezi.LA import operator as op
import numpy as np
import matplotlib.pyplot as plt
from Jiezi.Physics.common import ifdagger


def read_hamiltonian(path_hr):
    with open(path_hr, "r") as f:
        lines = f.readlines()
    num_wan = int(lines[1])
    num_rpts = int(lines[2])
    lines_dengeneracy = num_rpts // 15 + 1
    r_set = [list] * num_rpts
    hr_set = [None] * num_rpts
    for i in range(num_rpts):
        index_first_line = 3 + lines_dengeneracy + i * num_wan ** 2
        line_first = lines[index_first_line].split()
        r_set[i] = list(map(int, line_first[0: 3]))
        hamiltonian_R = matrix_numpy(num_wan, num_wan)
        for j in range(num_wan ** 2):
            line = lines[index_first_line + j].split()
            m = int(line[3]) - 1
            n = int(line[4]) - 1
            element_mn = float(line[5])
            hamiltonian_R.set_value(m, n, element_mn)
        hr_set[i] = hamiltonian_R
    # print(r_set)
    # print(h_set[19].get_value(69, 69))
    return r_set, hr_set


# define three basic lattice vectors in real space and get the lattice vector
def lattice_vector(r:list, r_1:vector_numpy, r_2:vector_numpy, r_3:vector_numpy):
    lat_vec = op.addvec(op.scamulvec(r[0], r_1), op.scamulvec(r[1], r_2), op.scamulvec(r[2], r_3))
    return lat_vec


def subband_k(hr_set, r_set, r_1, r_2, r_3, k_1, k_2, k_3, k):
    nn = hr_set[0].get_size()[0]
    hk = matrix_numpy(nn, nn)
    for i in range(len(r_set)):
        r_vec = lattice_vector(r_set[i], r_1, r_2, r_3)
        k_vec = lattice_vector(k, k_1, k_2, k_3)
        hk = op.addmat(hk, op.scamulmat(np.exp(1j * op.vecdotvec(k_vec.trans(), r_vec)), hr_set[i]))
    eigen_energy_k = hk.eigenvalue()
    return eigen_energy_k


path_hr = "/home/zjy/wannier90/wannier90-3.1.0/project/with_relax" + "/wannier90_hr.dat"
r_set, hr_set = read_hamiltonian(path_hr)

# test if hr_set[i]^dagger equals hr_set[2*10-i]
error = 0.0
for i in range(9):
    addition = op.addmat(hr_set[i], hr_set[2*10 - i])
    error += ifdagger(addition)
print(error)

# extract hamiltonian matrix of CNT
hr_cnt_set = [None] * len(hr_set)
for i in range(len(hr_set)):
    hr_cnt_set[i] = matrix_numpy()
    hr_temp = matrix_numpy()
    hr_temp.copy(hr_set[i].get_value())
    swap_list = [(4, 56), (5, 42), (13, 70), (15, 54), (23, 53), (35, 41)]
    for j in range(len(swap_list)):
        hr_temp = hr_temp.swap_index(swap_list[j][0], swap_list[j][1])
    ## swapped matrix
    hr_cnt_set[i].copy(hr_temp.get_value(40, 72, 40, 72))
    # hr_cnt_set[i].copy(hr_temp.get_value(0, 40, 0, 40))
    ## un-swapped matrix
    # hr_cnt_set[i].copy(hr_set[i].get_value(40, 72, 40, 72))
# print(hr_cnt_set[0].get_size())
# test if hr_set[i]^dagger equals hr_set[2*10-i]
error = 0.0
for i in range(9):
    addition = op.addmat(hr_cnt_set[i], hr_cnt_set[2*10 - i])
    error += ifdagger(addition)
print(error)


r_1 = vector_numpy(3)
r_2 = vector_numpy(3)
r_3 = vector_numpy(3)
r_1.set_value((0, 0), 24.6881123)
r_2.set_value((1, 0), 40)
r_3.set_value((2, 0), 4.2761526)
k_1 = vector_numpy(3)
k_2 = vector_numpy(3)
k_3 = vector_numpy(3)
k_1.set_value((0, 0), 2 * math.pi / 24.6881123)
k_2.set_value((1, 0), 2 * math.pi / 40)
k_3.set_value((2, 0), 2 * math.pi / 4.2761526)
k = [0, 0, -0.5]
eigen_energy_k = subband_k(hr_set, r_set, r_1, r_2, r_3, k_1, k_2, k_3, k)
print(eigen_energy_k.get_value())
print(eigen_energy_k.get_size())

num_kpts_cnt = 400
k_step = 0.01
k_start = 0
for i in range(num_kpts_cnt):
    k_cnt = [0, 0, k_step * i + k_start]
    kpt_plot = np.ones(hr_cnt_set[0].get_size()[0]) * k_cnt[2] * 2 * math.pi / 4.2761526
    subband_cnt_k = subband_k(hr_cnt_set, r_set, r_1, r_2, r_3, k_1, k_2, k_3, k_cnt)
    plt.scatter(kpt_plot, subband_cnt_k.get_value())
# plt.gca().set_aspect('equal', adjustable='box')
plt.show()
# path_band = "/home/zjy/wannier90/wannier90-3.1.0/project/with_relax" + "/wannier90_hr.dat"
# def band_structure(H: hamilton, start, end, step):
#     k_total = np.arange(start, end, step)
#     band = []
#     for k in k_total:
#         sub_band, U = subband(H, k)
#         band.append(sub_band[0])
#     return k_total, band


