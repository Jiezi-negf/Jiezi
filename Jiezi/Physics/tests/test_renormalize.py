# Copyright 2021 Jiezi authors.
#
# This file is part of Jiezi. It is subject to the license terms in the file
# LICENSE.md found in the top-level directory of this distribution. A list of
# Jiezi authors can be found in the file AUTHORS.md at the top-level directory
# of this distribution.
# ==============================================================================

import os
import sys

sys.path.append(os.path.abspath(__file__ + "/../../../.."))
from Jiezi.Physics.common import *
import numpy as np
import matplotlib.pyplot as plt
sys.path.append("../../../")

from Jiezi.Physics import hamilton
from Jiezi.Graph import builder

cnt = builder.CNT(n=4, m=0, Trepeat=6, nonideal=False)
cnt.construct()
H = hamilton.hamilton(cnt, onsite=-0.28, hopping=-2.97)
H.build_H()
H.build_S(base_overlap=0.018)
Hii_cell = H.get_Hii()[1]
Hi1_cell = H.get_Hi1()[1]
size = Hii_cell.get_size()[0]
x_init = hamilton.opValueInit2(Hii_cell, Hi1_cell, size)
L = 1
k_points = np.arange(-np.pi / L, np.pi / L, 0.05)
k_points_double = k_points / 2

# # ***** read the immediate optimized result Hii, Hi1 *****
# Hii_middle = matrix_numpy(size, size)
# Hi1_middle = matrix_numpy(size, size)
# with open('/home/zjy/Hii.txt', "r") as f:
#     lines = f.readlines()
#     tmp = np.asarray(lines).reshape((16, 4))
#     for i in range(16):
#         list_i = []
#         line = tmp[i]
#         list_i.append(list(map(float, line[0][2:-1].split()))[:])
#         list_i.append(list(map(float, line[1][1:-1].split()))[:])
#         list_i.append(list(map(float, line[2][1:-1].split()))[:])
#         list_i.append(list(map(float, line[3][1:-3].split()))[:])
#         Hii_middle.set_block_value(i, i+1, 0, 16, np.asarray(list_i).reshape(1, 16))
# with open('/home/zjy/Hi1.txt', "r") as f:
#     lines = f.readlines()
#     tmp = np.asarray(lines).reshape((16, 4))
#     for i in range(16):
#         list_i = []
#         line = tmp[i]
#         list_i.append(list(map(float, line[0][2:-1].split()))[:])
#         list_i.append(list(map(float, line[1][1:-1].split()))[:])
#         list_i.append(list(map(float, line[2][1:-1].split()))[:])
#         list_i.append(list(map(float, line[3][1:-3].split()))[:])
#         Hi1_middle.set_block_value(i, i+1, 0, 16, np.asarray(list_i).reshape(1, 16))


# # ***** plot band structure *****
# Hii_double, Hi1_double = hamilton.H_extendSize(Hii_cell, Hi1_cell, 2)
# Hii_opInit, Hi1_opInit = hamilton.transX2H(x_init, size)
# bandArray_cell = hamilton.compute_band(Hii_cell, Hi1_cell, L, k_points)
# bandArray_opInit = hamilton.compute_band(Hii_opInit, Hi1_opInit, L, k_points)
# bandArray_double = hamilton.compute_band(Hii_double, Hi1_double, 2 * L, k_points_double)
# plt.plot(k_points, bandArray_cell, color="green", label="original cell")
# plt.plot(k_points, bandArray_opInit, color="red", label="optimization started point")
# plt.plot(k_points_double, bandArray_double, color="black", label="double cell")
# plt.show()

# ***** renormalize optimization *****
# set initial value of optimization process
# x_init = hamilton.transH2X(Hii_cell, Hi1_cell)
# x_init = np.zeros(int((3 * size ** 2 + size) / 2))

# Hii_new, Hi1_new = hamilton.H_renormalize(Hii_cell, Hi1_cell, x_init, k_points, L)
# print(Hii_new.get_value(), Hi1_new.get_value())

# ***** renormalize based on sancho-rubio *****
eta = 5e-6
num_k = 100
num_E = 1000
k_list = np.linspace(-np.pi / L, np.pi / L, num_k)
E_list = np.linspace(-6, 6, num_E)
dos = np.zeros((num_k, num_E))

# plot dos of k and E which is computed from G(k, E)
for i, k in enumerate(k_list):
    for j, E in enumerate(E_list):
        dos[i, j] = hamilton.dosOfKE_SanchoRubio(Hii_cell, Hi1_cell, E, eta, k, 2 * L)
    # # plot E-K relationship on specific k point
    # eigenvalue_list = hamilton.compute_band(Hii_cell, Hi1_cell, L, list([k]))
    # plt.scatter(eigenvalue_list, np.zeros(eigenvalue_list.size))
    # # plot dos of E on specific E point
    # plt.plot(E_list, dos[i, :])
# plt.show()

# plot band structure
Hii_double, Hi1_double = hamilton.H_extendSize(Hii_cell, Hi1_cell, 2)
bandArray_double = hamilton.compute_band(Hii_double, Hi1_double, 2 * L, k_points)
plt.plot(k_points, bandArray_double, color="green", label="double cell")
plt.ylim(-6, 6)
x, y = np.meshgrid(k_list, E_list)
fig, ax = plt.subplots()
# c = ax.pcolormesh(x, y, dos.T, shading='gouraud', vmin=0, vmax=50)
c = ax.pcolormesh(x, y, dos.T)
fig.colorbar(c, ax=ax)
plt.show()


