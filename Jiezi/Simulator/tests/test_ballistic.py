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
from Jiezi.Physics.band import subband
from Jiezi.Physics.modespace import mode_space
from Jiezi.Physics.rgf_origin import rgf
# from Jiezi.Physics.rgf import rgf
from Jiezi.Physics.common import *
import numpy as np
from Jiezi.Physics import surface_gf, quantity
from matplotlib import pyplot as plt



from Jiezi.Physics import hamilton, band
from Jiezi.Graph import builder

def dos2np(dos, phi, E_list, mu):
    E_step = E_list[1] - E_list[0]
    if -phi < E_list[0]:
        zero_index = 0
    else:
        zero_index = int((-phi - E_list[0]) / E_step)
    n = np.trapz(dos[zero_index:] * fermi(E_list[zero_index:] - mu) * 2, dx=E_step)
    # n = np.trapz(dos[zero_index:] * (fermi(E_list[zero_index:]) + fermi(E_list[zero_index:] - mu)), dx=E_step)
    # p = np.trapz(dos[0:zero_index] * (1 - fermi(E_list - mu)), dx=E_step)
    return n

nz = 20
cnt = builder.CNT(n=8, m=0, Trepeat=nz, nonideal=False)
cnt.construct()
phi = 0
layer_phi_list = [phi] * nz
H = hamilton.hamilton(cnt, onsite=-phi, hopping=-2.97)
H.build_H()
H.build_S(base_overlap=0.018)

E_subband, U = subband(H, k=0)
# nm = 30
# Hii, Hi1, Sii, form_factor = mode_space(H, U, 16)
Hii = H.get_Hii()
Hi1 = H.get_Hi1()
Sii = H.get_Sii()
Si1 = H.get_Si1()
S00 = H.get_S00()
lead_H00_L, lead_H00_R = H.get_lead_H00()
lead_H10_L, lead_H10_R = H.get_lead_H10()
E_list = np.arange(-0.6601, 0.26, 0.01)
mul = 0
mur = 0
print("phi:", phi)
print("nz:", nz)
print("mur:", mur)

eta_rgf = 5e-6
eta_sg = 5e-6
volume_cell = 1
sigma_ph = []
nz = len(Hii)
nm = Hii[0].get_size()[0]
for i in range(len(E_list)):
    sigma_ph_ee = []
    for j in range(nz):
        sigma_ph_element = matrix_numpy(nm, nm)
        sigma_ph_ee.append(sigma_ph_element)
    sigma_ph.append(sigma_ph_ee)

# this is the output of rgf method

G_R_fullE = [None] * len(E_list)
G_lesser_fullE = [None] * len(E_list)
G_greater_fullE = [None] * len(E_list)
G1i_lesser_fullE = [None] * len(E_list)


Sigma_left_lesser_fullE = [None] * len(E_list)
Sigma_left_greater_fullE = [None] * len(E_list)
Sigma_right_lesser_fullE = [None] * len(E_list)
Sigma_right_greater_fullE = [None] * len(E_list)

for ee in range(len(E_list)):
    G_R_rgf, G_lesser, G_greater, G1i_lesser, Sigma_left_lesser, Sigma_left_greater, \
        Sigma_right_lesser, Sigma_right_greater = \
        rgf(ee, E_list, eta_rgf, mul, mur, Hii, Hi1, Sii, sigma_ph, sigma_ph)
    # G_R_rgf, G_lesser, G_greater, G1i_lesser, Sigma_left_lesser, Sigma_left_greater, \
    #     Sigma_right_lesser, Sigma_right_greater = \
    # rgf(ee, E_list, eta_rgf, mul, mur, Hii, Hi1, Sii, S00,
    #     lead_H00_L, lead_H00_R, lead_H10_L, lead_H10_R,
    #     sigma_ph, sigma_ph)


    # G_R_fullE, G_lesser_fullE, G_greater_fullE, G1i_lesser_fullE : [[], [], ...]
    # for example, length of G_lesser_fullE is len(E_list)
    # length of G_lesser_fullE[ee] is nz
    # G_lesser_fullE[ee][zz] is a matrix_numpy(nm, nm) object
    G_R_fullE[ee] = G_R_rgf
    G_lesser_fullE[ee] = G_lesser
    G_greater_fullE[ee] = G_greater
    G1i_lesser_fullE[ee] = G1i_lesser
    # Sigma_left_lesser_fullE, Sigma_left_greater_fullE: []
    # for example, length of Sigma_left_lesser_fullE is len(E_list)
    # Sigma_left_lesser_fullE[ee] is a matrix_numpy(nm, nm) object
    Sigma_left_lesser_fullE[ee] = Sigma_left_lesser
    Sigma_left_greater_fullE[ee] = Sigma_left_greater
    Sigma_right_lesser_fullE[ee] = Sigma_right_lesser
    Sigma_right_greater_fullE[ee] = Sigma_right_greater
n_spectrum_rgf, p_spectrum_rgf = quantity.carrierSpectrum(E_list, G_lesser_fullE, G_greater_fullE, volume_cell)
n_tol_rgf, p_tol_rgf = quantity.carrierQuantity(E_list, layer_phi_list, n_spectrum_rgf, p_spectrum_rgf)

print("ntol_rgf:", n_tol_rgf)
print("ptol_rgf:", p_tol_rgf)
dos_rgf = quantity.densityOfStates(E_list, G_R_fullE, volume_cell)
dos_rgf_np = np.array(dos_rgf)
n_L = dos2np(dos_rgf_np[:, 0], phi, E_list, mul)
n_R = dos2np(dos_rgf_np[:, nz-1], phi, E_list, mur)
print("rgf integral: n_L:", n_L)
print("rgf integral: n_R:", n_R)
print('\n')
print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<inv>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
# # <<<<<<<<<<<<<<<<<<< inverse directly <<<<<<<<<<<<<<<<<<<<<
# # construct the whole Hamiltonian matrix and the Sigma matrix
# H_total = matrix_numpy(nz * nm, nz * nm)
# for i in range(nz):
#     H_total.set_block_value(i * nm, (i + 1) * nm, i * nm, (i + 1) * nm, Hii[i])
#     if i > 0:
#         H_total.set_block_value((i - 1) * nm, i * nm, i * nm, (i + 1) * nm, Hi1[i])
#         H_total.set_block_value(i * nm, (i + 1) * nm, (i - 1) * nm, i * nm, Hi1[i].dagger())
# # construct the whole overlap matrix
# S_total = matrix_numpy(nz * nm, nz * nm)
# for i in range(nz):
#     S_total.set_block_value(i * nm, (i + 1) * nm, i * nm, (i + 1) * nm, Sii[i])
#     # if i > 0:
#     #     S_total.set_block_value((i - 1) * nm, i * nm, i * nm, (i + 1) * nm, Si1[i])
#     #     S_total.set_block_value(i * nm, (i + 1) * nm, (i - 1) * nm, i * nm, Si1[i].dagger())
#
# for ee in range(len(E_list)):
#     w = complex(E_list[ee], 0.0)
#     # compute the surface GF of left lead
#     G00_L = surface_gf.surface_gf(E_list[ee], eta_sg, Hii[0], Hi1[0].dagger(), Sii[0], TOL=1e-10)[0]
#     # compute the self energy of left lead based on the surface GF
#     Sigma_L = op.trimatmul(Hi1[0], G00_L, Hi1[0], type="cnn")
#     Gamma_L = op.scamulmat(complex(0.0, 1.0),
#                            op.addmat(Sigma_L, Sigma_L.dagger().nega()))
#     # Gamma_L = op.scamulmat(-2, Sigma_L.imaginary())
#     Sigma_lesser_L = op.scamulmat(fermi(E_list[ee] - mul), Gamma_L)
#     Sigma_greater_L = op.scamulmat(1.0 - fermi(E_list[ee] - mul), Gamma_L)
#
#     # compute the surface GF of right lead
#     G00_R = surface_gf.surface_gf(E_list[ee], eta_sg, Hii[nz - 1], Hi1[nz], Sii[nz - 1], TOL=1e-10)[0]
#     # compute the self energy of right lead based on the surface GF
#     Sigma_R = op.trimatmul(Hi1[nz], G00_R, Hi1[nz], type="nnc")
#     Gamma_R = op.scamulmat(complex(0.0, 1.0),
#                            op.addmat(Sigma_R, Sigma_R.dagger().nega()))
#     # Gamma_R = op.scamulmat(-2, Sigma_R.imaginary())
#     Sigma_lesser_R = op.scamulmat(fermi(E_list[ee] - mur), Gamma_R)
#     Sigma_greater_R = op.scamulmat(1.0 - fermi(E_list[ee] - mur), Gamma_R)
#
#     # construct the whole Sigma matrix and Sigma_lesser matrix
#     Sigma_total = matrix_numpy(nz * nm, nz * nm)
#     Sigma_total.set_block_value(0, nm, 0, nm, Sigma_L)
#     Sigma_total.set_block_value((nz - 1) * nm, nz * nm, (nz - 1) * nm, nz * nm, Sigma_R)
#
#     Gamma_L_total = matrix_numpy(nz * nm, nz * nm)
#     Gamma_R_total = matrix_numpy(nz * nm, nz * nm)
#     Gamma_L_total.set_block_value(0, nm, 0, nm, Gamma_L)
#     Gamma_R_total.set_block_value((nz - 1) * nm, nz * nm, (nz - 1) * nm, nz * nm, Gamma_R)
#
#     Sigma_lesser_total = matrix_numpy(nz * nm, nz * nm)
#     Sigma_lesser_total.set_block_value(0, nm, 0, nm, Sigma_lesser_L)
#     Sigma_lesser_total.set_block_value((nz - 1) * nm, nz * nm, (nz - 1) * nm, nz * nm, Sigma_lesser_R)
#
#     Sigma_greater_total = matrix_numpy(nz * nm, nz * nm)
#     Sigma_greater_total.set_block_value(0, nm, 0, nm, Sigma_greater_L)
#     Sigma_greater_total.set_block_value((nz - 1) * nm, nz * nm, (nz - 1) * nm, nz * nm, Sigma_greater_R)
#
#     # compute the GF of the whole system directly
#     G_R_inv = op.inv(op.addmat(op.scamulmat(w, S_total), H_total.nega(), Sigma_total.nega()))
#     G_lesser_inv = op.trimatmul(G_R_inv, Sigma_lesser_total, G_R_inv, type="nnc")
#     G_greater_inv = op.addmat(G_lesser_inv.nega(),
#                               op.scamulmat(complex(0.0, 1.0),
#                                            op.addmat(G_R_inv, G_R_inv.dagger().nega())))
#     G_R_inv_ee = [None] * nz
#     G_lesser_ee = [None] * nz
#     G_greater_ee = [None] * nz
#     for i in range(nz):
#         G_R_inv_ee_i = matrix_numpy(nm, nm)
#         G_lesser_ee_i = matrix_numpy(nm, nm)
#         G_greater_ee_i = matrix_numpy(nm, nm)
#         G_R_inv_ee_i.copy(G_R_inv.get_value(i * nm, (i + 1) * nm, i * nm, (i + 1) * nm))
#         G_lesser_ee_i.copy(G_lesser_inv.get_value(i * nm, (i + 1) * nm, i * nm, (i + 1) * nm))
#         G_greater_ee_i.copy(G_greater_inv.get_value(i * nm, (i + 1) * nm, i * nm, (i + 1) * nm))
#         G_R_inv_ee[i] = G_R_inv_ee_i
#         G_lesser_ee[i] = G_lesser_ee_i
#         G_greater_ee[i] = G_greater_ee_i
#     G_R_fullE[ee] = G_R_inv_ee
#     G_lesser_fullE[ee] = G_lesser_ee
#     G_greater_fullE[ee] = G_greater_ee
# n_spectrum_inv, p_spectrum_inv = quantity.carrierSpectrum(E_list, G_lesser_fullE, G_greater_fullE, volume_cell)
# n_tol_inv, p_tol_inv = quantity.carrierQuantity(E_list, layer_phi_list, n_spectrum_inv, p_spectrum_inv)
# print("ntol_inv:", n_tol_inv)
# print("ptol_inv:", p_tol_inv)
# dos_inv = quantity.densityOfStates(E_list, G_R_fullE, volume_cell)
# dos_inv_np = np.array(dos_inv)
# n_L = dos2np(dos_inv_np[:, 0], phi, E_list, mul)
# n_R = dos2np(dos_inv_np[:, nz-1], phi, E_list, mur)
# print("inv integral: n_L:", n_L)
# print("inv integral: n_R:", n_R)
# num_energy = len(E_list)
# error_G_lesser = [None] * num_energy
# error_G_greater = [None] * num_energy
# error_dos = [None] * num_energy
# for ee in range(num_energy):
#     error_G_lesser[ee] = np.absolute(np.array(n_spectrum_rgf[ee] / np.array(n_spectrum_inv[ee])  - np.ones(nz))).sum()
#     error_G_greater[ee] = np.absolute(np.array(p_spectrum_rgf[ee]) / np.array(p_spectrum_inv[ee] - np.ones(nz))).sum()
#     error_dos[ee] = np.absolute(np.array(dos_rgf[ee]) / np.array(dos_inv[ee] - np.ones(nz))).sum()
# plt.plot(E_list, error_G_lesser, label="e_lesser")
# plt.plot(E_list, error_G_greater, label="e_greater")
# plt.plot(E_list, error_dos, label="e_dos")
# plt.legend()
# plt.show()

