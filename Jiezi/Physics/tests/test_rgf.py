# Copyright 2021 Jiezi authors.
#
# This file is part of Jiezi. It is subject to the license terms in the file
# LICENSE.md found in the top-level directory of this distribution. A list of
# Jiezi authors can be found in the file AUTHORS.md at the top-level directory
# of this distribution.
# ==============================================================================


import sys
sys.path.append("../../../")
from Jiezi.Physics.band import subband
from Jiezi.Physics.modespace import mode_space
from Jiezi.Physics.rgf_origin import rgf
from Jiezi.Physics.common import *
import numpy as np
from Jiezi.Physics import surface_gf
import matplotlib.pyplot as plt



from Jiezi.Physics import hamilton, band
from Jiezi.Graph import builder

cnt = builder.CNT(n=8, m=0, Trepeat=3, nonideal=False)
cnt.construct()
phi = 0.1
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
E_list = [-0.0]
mul = 0.0
mur = -0.4
ee = 0
eta_rgf = 5e-6
eta_sg = 5e-6

w = complex(E_list[ee], 0.0)
sigma_ph = []
nz = len(Hii)
nm = Hii[0].get_size()[0]
for i in range(len(E_list)):
    sigma_ph_ee = []
    for j in range(nz):
        sigma_ph_element = matrix_numpy(nm, nm)
        sigma_ph_ee.append(sigma_ph_element)
    sigma_ph.append(sigma_ph_ee)


# G_R_rgf, G_lesser, G_greater, G1i_lesser, \
#             Sigma_left_lesser, Sigma_left_greate, Sigma_right_lesser, Sigma_right_greater = \
#                 rgf(ee, E_list, eta_rgf, mul, mur, Hii, Hi1, Sii, S00,
#                     lead_H00_L, lead_H00_R, lead_H10_L, lead_H10_R,
#                     sigma_ph, sigma_ph)

# compute the surface GF of left lead
G00_L = surface_gf.surface_gf(E_list[ee], eta_sg, Hii[0], Hi1[0].dagger(), Sii[0], TOL=1e-10)[0]
# compute the self energy of left lead based on the surface GF
Sigma_L = op.trimatmul(Hi1[0], G00_L, Hi1[0], type="cnn")
Gamma_L = op.scamulmat(complex(0.0, 1.0),
                       op.addmat(Sigma_L, Sigma_L.dagger().nega()))
Sigma_lesser_L = op.scamulmat(fermi(E_list[ee] - mul), Gamma_L)
Sigma_greater_L = op.scamulmat(1.0 - fermi(E_list[ee] - mul), Gamma_L)

# compute the surface GF of right lead
G00_R = surface_gf.surface_gf(E_list[ee], eta_sg, Hii[nz - 1], Hi1[nz], Sii[nz - 1], TOL=1e-10)[0]
# compute the self energy of right lead based on the surface GF
Sigma_R = op.trimatmul(Hi1[nz], G00_R, Hi1[nz], type="nnc")
Gamma_R = op.scamulmat(complex(0.0, 1.0),
                       op.addmat(Sigma_R, Sigma_R.dagger().nega()))
Sigma_lesser_R = op.scamulmat(fermi(E_list[ee] - mur), Gamma_R)
Sigma_greater_R = op.scamulmat(1.0 - fermi(E_list[ee] - mur), Gamma_R)

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
    # if i > 0:
    #     S_total.set_block_value((i - 1) * nm, i * nm, i * nm, (i + 1) * nm, Si1[i])
    #     S_total.set_block_value(i * nm, (i + 1) * nm, (i - 1) * nm, i * nm, Si1[i].dagger())

# construct the whole Sigma matrix and Sigma_lesser matrix
Sigma_total = matrix_numpy(nz * nm, nz * nm)
Sigma_total.set_block_value(0, nm, 0, nm, Sigma_L)
Sigma_total.set_block_value((nz - 1) * nm, nz * nm, (nz - 1) * nm, nz * nm, Sigma_R)

Gamma_L_total = matrix_numpy(nz * nm, nz * nm)
Gamma_R_total = matrix_numpy(nz * nm, nz * nm)
Gamma_L_total.set_block_value(0, nm, 0, nm, Gamma_L)
Gamma_R_total.set_block_value((nz - 1) * nm, nz * nm, (nz - 1) * nm, nz * nm, Gamma_R)

Sigma_lesser_total = matrix_numpy(nz * nm, nz * nm)
Sigma_lesser_total.set_block_value(0, nm, 0, nm, Sigma_lesser_L)
Sigma_lesser_total.set_block_value((nz - 1) * nm, nz * nm, (nz - 1) * nm, nz * nm, Sigma_lesser_R)

Sigma_greater_total = matrix_numpy(nz * nm, nz * nm)
Sigma_greater_total.set_block_value(0, nm, 0, nm, Sigma_greater_L)
Sigma_greater_total.set_block_value((nz - 1) * nm, nz * nm, (nz - 1) * nm, nz * nm, Sigma_greater_R)

# compute the GF of the whole system directly
G_R_inv = op.inv(op.addmat(op.scamulmat(w, S_total), H_total.nega(), Sigma_total.nega()))
G_R_inv_imag = G_R_inv.imaginary()
G_lesser_inv = op.trimatmul(G_R_inv, Sigma_lesser_total, G_R_inv, type="nnc")
G_greater_inv = op.addmat(G_lesser_inv.nega(),
                          op.scamulmat(complex(0.0, 1.0),
                                       op.addmat(G_R_inv, G_R_inv.dagger().nega())))

# this is the output of rgf method
G_R_rgf, G_lesser, G_greater, G1i_lesser, Sigma_left_lesser, Sigma_left_greater, \
    Sigma_right_lesser, Sigma_right_greater = \
    rgf(ee, E_list, eta_rgf, mul, mur, Hii, Hi1, Sii, sigma_ph, sigma_ph)


# only reserve the diagonal block, set the other blocks to be zero
G_R_inv_total = matrix_numpy(nz * nm, nz * nm)
for i in range(nz):
    G_R_inv_total.set_block_value(i * nm, (i + 1) * nm, i * nm, (i + 1) * nm,
                                  G_R_inv.get_value(i * nm, (i + 1) * nm, i * nm, (i + 1) * nm))

# transfer the block matrix G_R_rgf to the whole matrix G_R_rgf_total
G_R_rgf_total = matrix_numpy(nz * nm, nz * nm)
for i in range(nz):
    G_R_rgf_total.set_block_value(i * nm, (i + 1) * nm, i * nm, (i + 1) * nm, G_R_rgf[i])

G_R_rgf_total_imag = G_R_rgf_total.imaginary()
# compute the difference between the result got from two methods
delta_G_R = op.addmat(G_R_inv_total, G_R_rgf_total.nega())

# test if the current computed by inversion method is conserved
J_inv = [complex] * (nz + 1)
G_greater_inv_temp = matrix_numpy()
G_greater_inv_temp.copy(G_greater_inv.get_value(0, nm, 0, nm))
G_lesser_inv_temp = matrix_numpy()
G_lesser_inv_temp.copy(G_lesser_inv.get_value(0, nm, 0, nm))
J_L = op.addmat(
    op.matmulmat(G_greater_inv_temp, Sigma_lesser_L),
    op.matmulmat(G_lesser_inv_temp, Sigma_greater_L).nega()
).tre()
J_inv[0] = J_L.real
for i in range(0, nz - 1):
    Gi1_lesser_inv = matrix_numpy()
    Gi1_lesser_inv.copy(G_lesser_inv.get_value((i + 1) * nm, (i + 2) * nm, i * nm, (i + 1) * nm))
    J_inv[i + 1] = -2.0 * op.matmulmat(Hi1[i + 1], Gi1_lesser_inv).imaginary().tre()

G_greater_inv_temp.copy(G_greater_inv.get_value((nz-1)*nm, nz*nm, (nz-1)*nm, nz*nm))
G_lesser_inv_temp.copy(G_lesser_inv.get_value((nz-1)*nm, nz*nm, (nz-1)*nm, nz*nm))
J_R = - op.addmat(
    op.matmulmat(G_greater_inv_temp, Sigma_lesser_R),
    op.matmulmat(G_lesser_inv_temp, Sigma_greater_R).nega()
).tre()
J_inv[nz] = J_R.real
print(J_inv)

# compute current of coherent transport by transmission method
Transmission = op.matmulmat(op.trimatmul(Gamma_L_total, G_R_inv, Gamma_R_total, type="nnn"),
                            G_R_inv.dagger()).tre()
J_T = Transmission * (fermi(E_list[ee] - mul) - fermi(E_list[ee] - mur))
print("current computed by transmission formula is:", J_T)

# construct the G1i_lesser_rgf from the Gi1_lesser
# construct the Gi1_lesser_inv from the G_lesser_inv
# compute the difference between the two results
element_G1i_lesser_rgf = []
element_G1i_lesser_inv = []
element_delta_G1i = []
element_delta_G1i_imag = []
for i in range(len(G1i_lesser)):
    for j in range(nm):
        for k in range(nm):
            element_G1i_lesser_rgf.append(
                np.sqrt(G1i_lesser[i].get_value(j, k).real ** 2 + G1i_lesser[i].get_value(j, k).imag ** 2))
            element_G1i_lesser_inv.append(
                np.sqrt(G_lesser_inv.get_value((i + 1) * nm + j, i * nm + k).real ** 2 +
                        G_lesser_inv.get_value((i + 1) * nm + j, i * nm + k).imag ** 2)
            )
            element_delta_G1i.append(
                np.sqrt(
                    (G1i_lesser[i].get_value(j, k)-G_lesser_inv.get_value((i + 1) * nm + j, i * nm + k)).real**2 +
                    (G1i_lesser[i].get_value(j, k)-G_lesser_inv.get_value((i + 1) * nm + j, i * nm + k)).imag**2)
            )
            element_delta_G1i_imag.append(
                np.abs((G1i_lesser[i].get_value(j, k) - G_lesser_inv.get_value((i + 1) * nm + j, i * nm + k)).imag))


# test if matrix is conjugate
print("dagger test for H:", ifdagger(H_total))
print("dagger test for G_lesser_inv:", ifdagger(G_lesser_inv))

# visualize these data
element_delta_G_R = []
element_delta_G_R_imag = []
element_G_R_rgf = []
element_G_R_inv = []
for i in range(nm * nz):
    for j in range(nm * nz):
        element_G_R_rgf.append(
            np.sqrt(G_R_rgf_total.get_value(i, j).imag ** 2 + G_R_rgf_total.get_value(i, j).real ** 2))
        element_G_R_inv.append(np.sqrt(G_R_inv_total.get_value(i, j).imag ** 2 +
                                       G_R_inv_total.get_value(i, j).real ** 2))
        element_delta_G_R.append(np.sqrt(delta_G_R.get_value(i, j).imag ** 2 + delta_G_R.get_value(i, j).real ** 2))
        element_delta_G_R_imag.append(np.abs(delta_G_R.get_value(i, j).imag))

x = range((nm * nz) ** 2)
plt.subplot(1, 3, 1)
plt.title("rgf vs inversion: G_R")
plt.plot(x, element_delta_G_R, color="green", label="delta")
plt.plot(x, element_delta_G_R_imag, color="black", label="delta_imag")
# plt.plot(x, element_G_R_rgf, color="red", label="rgf")
# plt.plot(x, element_G_R_inv, color="blue", label="inv")
plt.legend()

y = range(nm ** 2 * (nz - 1))
plt.subplot(1, 3, 2)
plt.title("rgf vs inversion: G1i_lesser")
plt.plot(y, element_delta_G1i, color="green", label="delta")
plt.plot(y, element_delta_G1i_imag, color="black", label="delta_imag")
# plt.plot(y, element_G1i_lesser_rgf, color="red", label="rgf")
# plt.plot(y, element_G1i_lesser_inv, color="blue", label="inv")
plt.legend()

z = range(nz + 1)
plt.subplot(1, 3, 3)
plt.title("current")
plt.plot(z, J_inv, color="green", label="inv")
# plt.plot(y, element_G1i_lesser_rgf, color="red", label="rgf")
# plt.plot(y, element_G1i_lesser_inv, color="blue", label="inv")
plt.legend()
plt.show()
# print(element_delta)

# # plot the band
# k_total, band = band.band_structure(H, 0, 3 * 3.14 / 1.44, 1 * 3.14 / 1.44 / 20)
# print(band[0].get_value())
# i = 0
# for band_k in band:
#     k = np.ones(band[0].get_size()) * k_total[i]
#     i += 1
#     plt.scatter(k, band_k.get_value() / 2.97)
# plt.gca().set_aspect('equal', adjustable='box')
# plt.show()