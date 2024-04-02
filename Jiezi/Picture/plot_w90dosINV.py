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
from Jiezi.Physics.common import *
import numpy as np
from Jiezi.Physics import surface_gf
from Jiezi.Physics import hamilton, band
from Jiezi.Physics import w90_trans as w90
from Jiezi.Graph import builder
# import matplotlib.pyplot as plt
#
# def plotMatrix2D(matrix):
#     plt.imshow(np.absolute(matrix), cmap="inferno")
#     plt.colorbar()
#     plt.show()
# fig = plt.figure(figsize=(100,100))

def shift(num_cell, block_size_list, layerphi, Hmatrix):
    for i in range(num_cell):
        start = sum(block_size_list[0:i])
        end = sum(block_size_list[0:i + 1])
        nm = block_size_list[i]
        eye = matrix_numpy(nm, nm)
        eye.identity()
        phi = layerphi[i]
        matAdd = op.scamulmat(-phi, eye)
        matOrigin = matrix_numpy()
        matOrigin.copy(Hmatrix.get_value(start, end, start, end))
        matNow = op.addmat(matOrigin, matAdd)
        Hmatrix.set_block_value(start, end, start, end, matNow)

path_Files = os.path.abspath(os.path.join(__file__, "../..", "Files"))
path_hr = path_Files + "/wannier90_hr_new.dat"
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


cnt = builder.CNT(n=8, m=0, Trepeat=60, nonideal=False)
cnt.construct()
volume_cell = 530.9
H = hamilton.hamilton(cnt, onsite=0, hopping=-2.97)
H.build_H()
H.build_S(base_overlap=0.00)
H.H_readw90(Mii_total, Mi1_total, Mii_cnt, Mi1_cnt, num_supercell)
M_DL = matrix_numpy()
M_DL.copy(H.get_Hi1()[0].get_value())
M_DR = matrix_numpy()
M_DR.copy(H.get_Hi1()[-1].get_value())
Sii = matrix_numpy(Mii_total.get_size()[0], Mii_total.get_size()[0])
Sii.identity()


block_size_list = []
num_total_cell = 5
num_cnt_cell = 5
num_total_cell_2 = 5
num_cell = num_total_cell + num_cnt_cell + num_total_cell_2
size_total_cell = Mii_total.get_size()[0]
size_cnt_cell = Mii_cnt.get_size()[0]
for i in range(num_total_cell):
    block_size_list.append(size_total_cell)
for i in range(num_cnt_cell):
    block_size_list.append(size_cnt_cell)
for i in range(num_total_cell_2):
    block_size_list.append(size_total_cell)
size_H = sum(block_size_list)

H_total = matrix_numpy(size_H, size_H)

# left cnt+gra
for i in range(num_total_cell):
    nm = size_total_cell
    H_total.set_block_value(i * nm, (i + 1) * nm, i * nm, (i + 1) * nm, Mii_total)
    # plotMatrix2D(H_total.real().get_value())
    if i > 0:
        H_total.set_block_value((i - 1) * nm, i * nm, i * nm, (i + 1) * nm, Mi1_total)
        # plotMatrix2D(H_total.real().get_value())
        H_total.set_block_value(i * nm, (i + 1) * nm, (i - 1) * nm, i * nm, Mi1_total.dagger())
        # plotMatrix2D(H_total.real().get_value())
# device cnt
start = num_total_cell * size_total_cell
for i in range(num_cnt_cell):
    nm = size_cnt_cell
    H_total.set_block_value(start + i * nm, start + (i + 1) * nm, start + i * nm, start + (i + 1) * nm, Mii_cnt)
    # plotMatrix2D(H_total.real().get_value())
    if i > 0:
        H_total.set_block_value(start + (i - 1) * nm, start + i * nm, start + i * nm, start + (i + 1) * nm, Mi1_cnt)
        # plotMatrix2D(H_total.real().get_value())
        H_total.set_block_value(start + i * nm, start + (i + 1) * nm, start + (i - 1) * nm, start + i * nm,
                                Mi1_cnt.dagger())
        # plotMatrix2D(H_total.real().get_value())
# coupling between left and device
H_total.set_block_value((num_total_cell - 1) * size_total_cell, num_total_cell * size_total_cell,
                        num_total_cell * size_total_cell, num_total_cell * size_total_cell + size_cnt_cell, M_DL)
# plotMatrix2D(H_total.real().get_value())
H_total.set_block_value(num_total_cell * size_total_cell, num_total_cell * size_total_cell + size_cnt_cell,
                        (num_total_cell - 1) * size_total_cell, num_total_cell * size_total_cell, M_DL.dagger())
# plotMatrix2D(H_total.real().get_value())
# right cnt+gra
start = sum(block_size_list[0:num_total_cell + num_cnt_cell])
for i in range(num_total_cell_2):
    nm = size_total_cell
    H_total.set_block_value(start + i * nm, start + (i + 1) * nm, start + i * nm, start + (i + 1) * nm, Mii_total)
    # plotMatrix2D(H_total.real().get_value())
    if i > 0:
        H_total.set_block_value(start + (i - 1) * nm, start + i * nm, start + i * nm, start + (i + 1) * nm, Mi1_total)
        # plotMatrix2D(H_total.real().get_value())
        H_total.set_block_value(start + i * nm, start + (i + 1) * nm, start + (i - 1) * nm, start + i * nm,
                                Mi1_total.dagger())
        # plotMatrix2D(H_total.real().get_value())
# coupling between right and device
H_total.set_block_value(start - size_cnt_cell, start, start, start + size_total_cell, M_DR)
# plotMatrix2D(H_total.real().get_value())
H_total.set_block_value(start, start + size_total_cell, start - size_cnt_cell, start, M_DR.dagger())
# plotMatrix2D(H_total.real().get_value())
layerphi = [0.4261241881782552, 0.4231241881782552, 0.4231241881782552, 0.4261241881782552,
            0.4261241881782552, 0.4231241881782552, 0.4231241881782552, 0.4231241881782552,
            0.4231241881782552, 0.4053786112852024, 0.4053786112852024, 0.4053786112852024,
            0.39, 0.39, 0.38]
shift(num_cell, block_size_list, layerphi, H_total)

num_E = 300
E_list = np.linspace(-1.6605, 1.26, num_E)
eta_sg = 5e-6

dos = np.zeros((num_E * num_cell, 3))

for ee in range(num_E):
    print(ee)
    dos[ee * num_cell: (ee + 1) * num_cell, 0] = np.asarray([E_list[ee]] * num_cell).reshape((num_cell, 1))[:, 0]
    w = complex(E_list[ee], 0.0)
    # compute the surface GF of left lead
    G00_L = surface_gf.surface_gf(E_list[ee], eta_sg, Mii_total, Mi1_total.dagger(), Sii, TOL=1e-10)[0]
    # compute the self energy of left lead based on the surface GF
    Sigma_L = op.trimatmul(Mi1_total, G00_L, Mi1_total, type="cnn")

    # compute the surface GF of right lead
    G00_R = surface_gf.surface_gf(E_list[ee], eta_sg, Mii_total, Mi1_total, Sii, TOL=1e-10)[0]
    # compute the self energy of right lead based on the surface GF
    Sigma_R = op.trimatmul(Mi1_total, G00_R, Mi1_total, type="nnc")

    # construct the whole overlap matrix
    S_total = matrix_numpy(size_H, size_H)
    S_total.identity()


    # construct the whole Sigma matrix and Sigma_lesser matrix
    Sigma_total = matrix_numpy(size_H, size_H)
    Sigma_total.set_block_value(0, block_size_list[0], 0, block_size_list[0], Sigma_L)
    Sigma_total.set_block_value(size_H - size_total_cell, size_H, size_H - size_total_cell, size_H, Sigma_R)


    # compute the GF of the whole system directly
    G_R_inv_total = op.inv(op.addmat(op.scamulmat(w, S_total), H_total.nega(), Sigma_total.nega()))

    for i in range(num_cell):
        z = length_single_cell * (i + 0.5) * num_supercell
        dos[ee * num_cell + i, 1] = z
        start = sum(block_size_list[0:i])
        nm = block_size_list[i]
        dos[ee * num_cell + i, 2] = (- G_R_inv_total.get_value(start, start + nm, start, start + nm).trace().imag /
                                  volume_cell / math.pi)


path_Files = os.path.abspath(os.path.join(__file__, "../..", "Files"))
fileName = "/w90dosINV.dat"
fname = path_Files + fileName
np.savetxt(fname, dos, fmt='%.18e', delimiter=' ', newline='\n')

#
# import matplotlib.pyplot as plt
# path_Files = os.path.abspath(os.path.join(__file__, "../..", "Files"))
# fileName = "/w90dosINV.dat"
# fname = path_Files + fileName
# dos = np.loadtxt(fname)
# num_E = 300
# num_cell = 15
# x = np.linspace(-1.6605, 1.26, num_E)
# y = np.zeros((num_cell, num_E))
# for i in range(num_cell):
#     for j in range(num_E):
#         if i <= 5 or i >=10:
#             y[i, j] = min(0.04, dos[j * num_cell + i, 2])
#         else:
#             y[i, j] = min(0.04, dos[j * num_cell + i, 2] / 4 * 9)
#     plt.plot(x, y[i, :], label=str(i))
# plt.legend()
# plt.show()
# data = np.zeros((num_cell * num_E, 3))
# for ee in range(num_E):
#     for i in range(num_cell):
#         z = 4.27 * (i + 0.5) * 2
#         data[ee * num_cell + i, 0] = x[ee]
#         data[ee * num_cell + i, 1] = z
#         if i <= 4 or i >= 10:
#             data[ee * num_cell + i, 2] = min(0.04, dos[ee * num_cell + i, 2])
#         else:
#             data[ee * num_cell + i, 2] = min(0.04, dos[ee * num_cell + i, 2] / 4 * 9)
# np.savetxt(path_Files+'/postW90dosINV.dat', data, fmt='%.18e', delimiter=' ', newline='\n')
#
#
# plotPhi = np.zeros((num_cell, 3))
# layerphi = [0.4261241881782552, 0.4231241881782552, 0.4231241881782552, 0.4261241881782552,
#             0.4261241881782552, 0.4231241881782552, 0.4231241881782552, 0.4231241881782552,
#             0.4231241881782552, 0.4053786112852024, 0.4053786112852024, 0.4053786112852024,
#             0.39, 0.39, 0.38]
# for i in range(num_cell):
#     z = 4.27 * (i + 0.5) * 2
#     plotPhi[i, 0] = z
#     plotPhi[i, 1] = 0.276 - layerphi[i]
#     plotPhi[i, 2] = - 0.276 - layerphi[i]
# np.savetxt(path_Files+'/postW90PhiINV.dat', plotPhi, fmt='%.18e', delimiter=' ', newline='\n')
