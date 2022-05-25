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

cnt = builder.CNT(n=5, m=5, Trepeat=3, nonideal=False)
cnt.construct()
H = hamilton.hamilton(cnt, onsite=-0.28, hopping=-2.97)
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
start = min(mul, mur, min_subband)-0.5
end = max(mul, mur, max_subband)+0.5
step = 0.01
E_list = np.arange(start, end, step)

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

    # compute the surface GF of right lead
    G00_R = surface_gf.surface_gf(E_list[ee], eta_sg, Hii[nz - 1], Hi1[nz], Sii[nz - 1], TOL=1e-100)[0]
    # compute the self energy of right lead based on the surface GF
    Sigma_R = op.trimatmul(Hi1[nz], G00_R, Hi1[nz], type="nnc")

    # construct the whole Sigma matrix
    Sigma_total = matrix_numpy(nz * nm, nz * nm)
    Sigma_total.set_block_value(0, nm, 0, nm, Sigma_L)
    Sigma_total.set_block_value((nz - 1) * nm, nz * nm, (nz - 1) * nm, nz * nm, Sigma_R)

    # compute the GF of the whole system directly
    G_R_inv = op.inv(op.addmat(op.scamulmat(w, S_total), H_total.nega(), Sigma_total.nega()))

    # this is the output of rgf method
    # G_R_rgf, G_lesser, G_greater, G1i_lesser, Sigma_left_lesser, Sigma_left_greater, \
    #     Sigma_right_lesser, Sigma_right_greater = \
    #     rgf(ee, E_list, eta, mul, mur, Hii, Hi1, Sii, sigma_ph, sigma_ph)

    # compute the density of state DOS(E,x) by inv method
    # for z in range(nz):
    #     tre_z = 0
    #     for nn in range(nm):
    #         tre_z += - G_R_inv.get_value(z * nm + nn, z * nm + nn).imag
    #     axis_energy.append(E_list[ee])
    #     axis_z.append(z)
    #     axis_dos_inv.append(tre_z)
    for z in range(nz):
        tre = 0.0
        for nn in range(nm):
            tre += - G_R_inv.get_value(z * nm + nn, z * nm + nn).imag
        dos_inv[z, ee] = tre

    # compute the density of state DOS(E,x) by rgf method
    # for z in range(nz):
    #     axis_dos_rgf.append(- G_R_rgf[z].tre().imag)

# visualize
fig, ax1 = plt.subplots()
k_total, band = band.band_structure(H, 0.0, 3 * 3.14 / 1.44, 1 * 3.14 / 1.44 / 80)
print(band[0].get_value())
i = 0
for band_k in band:
    k = np.ones(band[0].get_size()) * k_total[i]
    i += 1
    ax1.scatter(k, band_k.get_value(), s=10, color="red")
# plt.gca().set_aspect('equal', adjustable='box')

ax2 = ax1.twiny()
# ax2.invert_xaxis()
ax2.plot(dos_inv[1, :], E_list, color='green', label="0")
plt.legend()
plt.show()

# axis_energy = np.array(axis_energy)
# axis_z = np.array(axis_z)
# axis_dos_inv = np.array(axis_dos_inv)
# fig = plt.figure()
# ax = Axes3D(fig)
# # axis_energy, axis_z = np.meshgrid(axis_energy, axis_z)
# ax.plot_trisurf(axis_energy, axis_z, axis_dos_inv, cmap='rainbow')


# plt.subplot(1, 2, 2)
# plt.plot(E_list, dos_inv[0, :], color='green', label="0")
# plt.plot(E_list, dos_inv[1, :], color='blue', label="1")
# plt.plot(E_list, dos_inv[2, :], color='red', label="2")
# plt.legend()
# plt.show()