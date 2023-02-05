# Copyright 2021 Jiezi authors.
#
# This file is part of Jiezi. It is subject to the license terms in the file
# LICENSE.md found in the top-level directory of this distribution. A list of
# Jiezi authors can be found in the file AUTHORS.md at the top-level directory
# of this distribution.
# ==============================================================================


import sys
import matplotlib.pyplot as plt
from Jiezi.Graph import builder
from Jiezi.NEGF.tests.fake_potential import fake_potential
from Jiezi.Physics import hamilton
from Jiezi.Physics.band import subband
from Jiezi.Physics.modespace import mode_space
from Jiezi.Physics.rgf import rgf
from Jiezi.LA.matrix_numpy import matrix_numpy
from Jiezi.LA.base_linalg import matrix
from Jiezi.LA import operator as op
from Jiezi.Physics.common import *
from Jiezi.Physics.quantity import quantity
import numpy as np
from Jiezi.Physics import surface_gf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

sys.path.append("../../../")

from Jiezi.Physics import hamilton, band
from Jiezi.Graph import builder

cnt = builder.CNT(n=4, m=0, Trepeat=3, nonideal=False)
cnt.construct()
phi = 1.0
H = hamilton.hamilton(cnt, onsite=-phi-0.28, hopping=-2.97)
H.build_H()
H.build_S(base_overlap=0.018)

E_subband, U = subband(H, k=0)
# nm = 30
# Hii, Hi1, Sii, form_factor = mode_space(H, U, 16)
Hii = H.get_Hii()
Hi1 = H.get_Hi1()
Sii = H.get_Sii()
Si1 = H.get_Si1()

eta_sg = 5e-6
eta_gf = 0.0
nz = len(Hii)
nm = Hii[0].get_size()[0]

# pick up the min and max value of E_subband
min_temp = []
max_temp = []
for energy in E_subband:
    min_temp.append(energy.get_value()[0])
    max_temp.append(energy.get_value()[nm - 1])
min_subband = min(min_temp).real
max_subband = max(max_temp).real

# define Energy list that should be computed
start = min(mul, mur, min_subband) - 0.26
end = max(mul, mur, max_subband) + 0.26
step = 0.05
E_list = np.arange(start, end, step)
print("E_start:", start, "E_end:", end)

# construct the whole Hamiltonian matrix and the Sigma matrix
H_total = matrix_numpy(nz * nm, nz * nm)
for i in range(nz):
    H_total.set_block_value(i * nm, (i + 1) * nm, i * nm, (i + 1) * nm, Hii[i])
    if i > 0:
        H_total.set_block_value((i - 1) * nm, i * nm, i * nm, (i + 1) * nm, Hi1[i])
        H_total.set_block_value(i * nm, (i + 1) * nm, (i - 1) * nm, i * nm, Hi1[i].dagger())

# construct the whole overlap matrix
S_total = matrix_numpy(nz * nm, nz * nm)
for i in range(nz):
    S_total.set_block_value(i * nm, (i + 1) * nm, i * nm, (i + 1) * nm, Sii[i])
    if i > 0:
        S_total.set_block_value((i - 1) * nm, i * nm, i * nm, (i + 1) * nm, Si1[i])
        S_total.set_block_value(i * nm, (i + 1) * nm, (i - 1) * nm, i * nm, Si1[i].dagger())

axis_energy = []
axis_z = []
axis_dos_inv = []
axis_dos_rgf = []
dos_inv = np.zeros((nz, len(E_list)))
dos_rgf = np.zeros((nz, len(E_list)))
G_lesser_0 = []
G_greater_0 = []
G_lplusg_0 = []
G_lesser_inv_list = np.zeros((len(E_list), nz))
G_greater_inv_list = np.zeros((len(E_list), nz))

for ee in range(len(E_list)):
    sigma_ph = []
    w = complex(E_list[ee], eta_gf)
    for i in range(len(E_list)):
        sigma_ph_ee = []
        for j in range(nz):
            sigma_ph_element = matrix_numpy(nm, nm)
            sigma_ph_ee.append(sigma_ph_element)
        sigma_ph.append(sigma_ph_ee)

    # compute the surface GF of left lead
    G00_L = surface_gf.surface_gf(E_list[ee], eta_sg, Hii[0], Hi1[0].dagger(), Sii[0], TOL=1e-100)[0]
    # compute the self energy of left lead based on the surface GF
    Sigma_L = op.trimatmul(Hi1[0], G00_L, Hi1[0], type="cnn")
    Sigma_L_lesser = op.scamulmat(-2 * fermi(E_list[ee] - mul), Sigma_L.imaginary())
    Sigma_L_greater = op.scamulmat(-2 * (1 - fermi(E_list[ee] - mul)), Sigma_L.imaginary())
    # Sigma_L_lesser = op.scamulmat(-2j * fermi(E_list[ee] - mul), Sigma_L.imaginary())
    # Sigma_L_greater = op.scamulmat(2j * (1 - fermi(E_list[ee] - mul)), Sigma_L.imaginary())

    # compute the surface GF of right lead
    G00_R = surface_gf.surface_gf(E_list[ee], eta_sg, Hii[nz - 1], Hi1[nz], Sii[nz - 1], TOL=1e-100)[0]
    # compute the self energy of right lead based on the surface GF
    Sigma_R = op.trimatmul(Hi1[nz], G00_R, Hi1[nz], type="nnc")
    Sigma_R_lesser = op.scamulmat(-2 * fermi(E_list[ee] - mur), Sigma_R.imaginary())
    Sigma_R_greater = op.scamulmat(-2 * (1 - fermi(E_list[ee] - mur)), Sigma_R.imaginary())
    # Sigma_R_lesser = op.scamulmat(-2j * fermi(E_list[ee] - mur), Sigma_R.imaginary())
    # Sigma_R_greater = op.scamulmat(2j * (1 - fermi(E_list[ee] - mur)), Sigma_R.imaginary())

    # construct the whole Sigma matrix
    Sigma_total = matrix_numpy(nz * nm, nz * nm)
    Sigma_total.set_block_value(0, nm, 0, nm, Sigma_L)
    Sigma_total.set_block_value((nz - 1) * nm, nz * nm, (nz - 1) * nm, nz * nm, Sigma_R)

    # construct the whole Sigma_lesser and Sigma_greater matrix
    Sigma_lesser_total = matrix_numpy(nz * nm, nz * nm)
    Sigma_greater_total = matrix_numpy(nz * nm, nz * nm)
    Sigma_lesser_total.set_block_value(0, nm, 0, nm, Sigma_L_lesser)
    Sigma_lesser_total.set_block_value((nz - 1) * nm, nz * nm, (nz - 1) * nm, nz * nm, Sigma_R_lesser)
    Sigma_greater_total.set_block_value(0, nm, 0, nm, Sigma_L_greater)
    Sigma_greater_total.set_block_value((nz - 1) * nm, nz * nm, (nz - 1) * nm, nz * nm, Sigma_R_greater)

    # compute the GF of the whole system directly
    G_R_inv = op.inv(op.addmat(op.scamulmat(w, S_total), H_total.nega(), Sigma_total.nega()))
    G_lesser_inv = op.trimatmul(G_R_inv, Sigma_lesser_total, G_R_inv, type="nnc")
    G_greater_inv = op.trimatmul(G_R_inv, Sigma_greater_total, G_R_inv, type="nnc")

    # compute n and p by inv
    for zz in range(nz):
        G_lesser_inv_eezz = G_lesser_inv.get_value(zz * nm, (zz + 1) * nm, zz * nm, (zz + 1) * nm)
        G_lesser_inv_list[ee, zz] = G_lesser_inv_eezz.trace()
        G_greater_inv_eezz = G_greater_inv.get_value(zz * nm, (zz + 1) * nm, zz * nm, (zz + 1) * nm)
        G_greater_inv_list[ee, zz] = G_greater_inv_eezz.trace()
        # G_lesser_inv_eezz = G_lesser_inv.imaginary().get_value(zz * nm, (zz + 1) * nm, zz * nm, (zz + 1) * nm)
        # G_lesser_inv_list[ee, zz] = G_lesser_inv_eezz.trace()
        # G_greater_inv_eezz = G_greater_inv.imaginary().get_value(zz * nm, (zz + 1) * nm, zz * nm, (zz + 1) * nm)
        # G_greater_inv_list[ee, zz] = G_greater_inv_eezz.trace()

n_inv = np.zeros(nz)
p_inv = np.zeros(nz)
for zz in range(nz):
    for ee in range(len(E_list) - 1):
        if E_list[ee] > -phi:
            n_inv[zz] += step / 2 * (G_lesser_inv_list[ee, zz] + G_lesser_inv_list[ee + 1, zz])
        if E_list[ee] < -phi:
            p_inv[zz] += step / 2 * (G_greater_inv_list[ee, zz] + G_greater_inv_list[ee + 1, zz])
print("n_inv:", n_inv)
print("p_inv:", p_inv)
plt.plot(np.arange(len(n_inv)), n_inv)
plt.show()


#     # # this is the output of rgf method
#     # G_R_rgf, G_lesser, G_greater, G1i_lesser, Sigma_left_lesser, Sigma_left_greater, \
#     #     Sigma_right_lesser, Sigma_right_greater = \
#     #     rgf(ee, E_list, eta_sg, mul, mur, Hii, Hi1, Sii, sigma_ph, sigma_ph)
#
#     # # compute the difference between the G_lesser/G_greater from inverse method and rgf method
#     # for z in range(nz):
#     #     diff_G_lesser = (G_lesser_inv.get_value(z * nm, (z + 1) * nm, z * nm, (z + 1) * nm) -
#     #                      G_lesser[z].get_value()).sum()
#     #     diff_G_greater = (G_greater_inv.get_value(z * nm, (z + 1) * nm, z * nm, (z + 1) * nm) -
#     #                       G_greater[z].get_value()).sum()
#     #     print("energy:", E_list[ee], "cell_index:", z, "the G_lesser difference between inv and rgf is:",
#     #           diff_G_lesser)
#     #     print("energy:", E_list[ee], "cell_index:", z, "the G_greater difference between inv and rgf is:",
#     #           diff_G_greater)
#     #     diff_G_lesser_element = []
#     #     G_lesser_inv_element = []
#     #     G_lesser_rgf_element = []
#     #     for i in range(nm):
#     #         for j in range(nm):
#     #             G_lesser_inv_element_ij = G_lesser_inv.get_value(z * nm + i, z * nm + j)
#     #             G_lesser_rgf_element_ij = G_lesser[z].get_value(i, j)
#     #             G_lesser_inv_element.append(G_lesser_inv_element_ij)
#     #             G_lesser_rgf_element.append(G_lesser_rgf_element_ij)
#     #             diff_G_lesser_element.append(abs(G_lesser_inv_element_ij - G_lesser_rgf_element_ij))
#     #     plt.plot(np.arange(0, nm * nm, 1), G_lesser_inv_element, color='blue', label="G_lesser_inv_element")
#     #     plt.plot(np.arange(0, nm * nm, 1), G_lesser_rgf_element, color='red', label="G_lesser_rgf_element")
#     #     plt.plot(np.arange(0, nm * nm, 1), diff_G_lesser_element, color='black', label="diff_G_lesser_element")
#     #     plt.show()
#     # compute the density of state DOS(E,x) by inv method
#     # for z in range(nz):
#     #     tre_z = 0
#     #     for nn in range(nm):
#     #         tre_z += - G_R_inv.get_value(z * nm + nn, z * nm + nn).imag
#     #     axis_energy.append(E_list[ee])
#     #     axis_z.append(z)
#     #     axis_dos_inv.append(tre_z)
#
#     # for z in range(nz):
#     #     tre = 0.0
#     #     for nn in range(nm):
#     #         tre += - G_R_inv.get_value(z * nm + nn, z * nm + nn).imag
#     #     dos_inv[z, ee] = tre
#
#     # compute the density of state DOS(E,x) by rgf method
#     for z in range(nz):
#         dos_rgf[z, ee] = - 2 * G_R_rgf[z].tre().imag
#
#     # store G_lesser[ee][0] to compute n
#     G_lesser_0.append(G_lesser[1].tre())
#     G_greater_0.append(G_greater[1].tre())
#     G_lplusg_0.append(G_lesser[1].tre() + G_greater[1].tre())
#
#
# # # visualize
# # fig, ax1 = plt.subplots()
# # k_total, band = band.band_structure(H, 0.0, 3 * 3.14 / 1.44, 1 * 3.14 / 1.44 / 80)
# # print(band[0].get_value())
# # i = 0
# # for band_k in band:
# #     k = np.ones(band[0].get_size()) * k_total[i]
# #     i += 1
# #     ax1.scatter(k, band_k.get_value(), s=10, color="red")
# # # plt.gca().set_aspect('equal', adjustable='box')
# #
# # ax2 = ax1.twiny()
# # # ax2.invert_xaxis()
# # ax2.plot(dos_inv[1, :], E_list, color='blue', label="inv_1")
# # ax2.plot(dos_rgf[1, :], E_list, color='green', label="rgf_1")
# # plt.legend()
# # plt.show()
#
# # axis_energy = np.array(axis_energy)
# # axis_z = np.array(axis_z)
# # axis_dos_inv = np.array(axis_dos_inv)
# # fig = plt.figure()
# # ax = Axes3D(fig)
# # # axis_energy, axis_z = np.meshgrid(axis_energy, axis_z)
# # ax.plot_trisurf(axis_energy, axis_z, axis_dos_inv, cmap='rainbow')
#
#
# # plt.subplot(1, 2, 2)
# # plt.plot(E_list, dos_inv[0, :], color='green', label="0")
# # plt.plot(E_list, dos_inv[1, :], color='blue', label="1")
# # plt.plot(E_list, dos_inv[2, :], color='red', label="2")
# # plt.legend()
# # plt.show()
#
# # compute n from dos and G^lesser
# n_dos = 0.0
# p_dos = 0.0
# n_g_lesser = 0
# p_g_greater = 0
# for ee in range(len(E_list) - 1):
#     if E_list[ee] > -phi:
#         n_dos += step / 2 * (dos_rgf[1, ee] * fermi(E_list[ee]) + dos_rgf[1, ee + 1] * fermi(E_list[ee + 1]))
#         # n_g_lesser += step / 2 * (G_lesser_0[ee] + G_lesser_0[ee + 1])
# print(n_dos)
# for ee in range(len(E_list) - 1):
#     if E_list[ee] > -phi:
#         n_g_lesser += step / 2 * (G_lesser_0[ee] + G_lesser_0[ee + 1])
# print(n_g_lesser)
#
# for ee in range(len(E_list) - 1):
#     if E_list[ee] < -phi:
#         p_dos += step / 2 * (dos_rgf[1, ee] * (1 - fermi(E_list[ee])) +
#                              dos_rgf[1, ee + 1] * (1 - fermi(E_list[ee + 1])))
#         # n_g_lesser += step / 2 * (G_lesser_0[ee] + G_lesser_0[ee + 1])
# print(p_dos)
#
# for ee in range(len(E_list) - 1):
#     if E_list[ee] < -phi:
#         p_g_greater += step / 2 * (G_greater_0[ee] + G_greater_0[ee + 1])
# print(p_g_greater)
#
# # plt.subplot(1, 2, 1)
# # plt.plot(E_list, G_lesser_0, color='green', label="e")
# # plt.plot(E_list, G_greater_0, color='blue', label="p")
# # plt.legend()
# # plt.subplot(1, 2, 2)
# # plt.plot(E_list, G_lplusg_0, color='yellow', label="e+p")
# # plt.plot(E_list, dos_rgf[0, :], color='black', label="dos_rgf")
# # plt.legend()
# # plt.show()