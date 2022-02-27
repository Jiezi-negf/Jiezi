# Copyright 2021 Jiezi authors.
#
# This file is part of Jiezi. It is subject to the license terms in the file
# LICENSE.md found in the top-level directory of this distribution. A list of
# Jiezi authors can be found in the file AUTHORS.md at the top-level directory
# of this distribution.
# ==============================================================================


import sys


import numpy as np
import matplotlib.pyplot as plt
sys.path.append("../../../")

from Jiezi.Physics import hamilton, band, modespace, surface_gf
from Jiezi.Graph import builder
from Jiezi.LA import operator as op


cnt = builder.CNT(n=5, m=5, Trepeat=3, nonideal=False)
cnt.construct()
H = hamilton.hamilton(cnt, onsite=-0.28, hopping=-2.97)
H.build_H()
H.build_S(base_overlap=0.018)
E_subband, U = band.subband(H, k=0)

E_list = [0]
Hii = H.get_Hii()
Hi1 = H.get_Hi1()
Sii = H.get_Sii()
nz = len(Hii)
nm = Hii[0].get_size()[0]
# Hii, Hi1, Sii, form_factor = modespace.mode_space(H, U, "low", nm)
ee = 0
eta = 5e-6
GBB_L = surface_gf.surface_gf(E_list[ee], eta, Hii[0], Hi1[0].dagger(), Sii[0], TOL=1e-12)[1]
GBB_R = surface_gf.surface_gf(E_list[ee], eta, Hii[nz - 1], Hi1[nz], Sii[nz - 1], TOL=1e-12)[1]
delta_GBB = op.addmat(GBB_L, GBB_R.nega())
# print(GBB_L.get_value())
# print(GBB_R.get_value())
# print(np.max(abs(op.addmat(GBB_L, GBB_R.nega()).get_value().real)))
# print(np.sum(abs(op.addmat(GBB_L, GBB_R.nega()).get_value().real)))
G00_L_ref = surface_gf.surface_gf(E_list[ee], eta, Hii[0], Hi1[0].dagger(), Sii[0], TOL=1e-12)[0]
# G00_L_dumb = surface_gf.surface_gf_dumb(E_list[ee], eta, Hii[0], Hi1[0].dagger(), Sii[0], TOL=1e-12)
G00_L_dumb = surface_gf.surface_gf(E_list[ee], eta, Hii[0], Hi1[0].dagger(), Sii[0], TOL=1e-12)[0]
row = G00_L_ref.get_size()[0]
column = G00_L_ref.get_size()[1]
sum = 0.0
delta = op.addmat(G00_L_ref, G00_L_dumb.nega())
# print("G00 is:", G00_L_dumb.get_value())
# for i in range(row):
#     for j in range(column):
#         sum += np.sqrt(delta.get_value(i, j).real ** 2 + delta.get_value(i, j).imag ** 2)
# print("delta sum is:", sum/(row * column))
# print("max delta element is:", np.max(delta.get_value()))

# visualize these data
element_clever = []
element_dumb = []
element_delta = []
element_GBB_L = []
element_GBB_R = []
element_delta_GBB = []
num_row, num_col = delta.get_size()
for i in range(num_row):
    for j in range(num_col):
        element_clever.append(np.sqrt(G00_L_ref.get_value(i, j).imag**2+G00_L_ref.get_value(i, j).real**2))
        element_dumb.append(np.sqrt(G00_L_dumb.get_value(i, j).imag ** 2 + G00_L_dumb.get_value(i, j).real ** 2))
        element_delta.append(np.sqrt(delta.get_value(i, j).imag ** 2 + delta.get_value(i, j).real ** 2))
        element_GBB_L.append(np.sqrt(GBB_L.get_value(i, j).imag ** 2 + GBB_L.get_value(i, j).real ** 2))
        element_GBB_R.append(np.sqrt(GBB_R.get_value(i, j).imag ** 2 + GBB_R.get_value(i, j).real ** 2))
        element_delta_GBB.append(np.sqrt(delta_GBB.get_value(i, j).imag ** 2 + delta_GBB.get_value(i, j).real ** 2))
x = range(num_row * num_col)
fig, axs = plt.subplots(1, 3)
axs[0].set_title("sancho-rubio vs dumb")
axs[0].plot(x, element_clever, color="red", label="sancho-rubio")
axs[0].plot(x, element_dumb, color="blue", label="dumb")
axs[0].plot(x, element_delta, color="green", label="delta")
axs[0].legend()

axs[1].set_title("GBB from left and right by sancho-rubio method")
axs[1].plot(x, element_GBB_L, color="red", label="GBB from left")
axs[1].plot(x, element_GBB_R, color="blue", label="GBB from right")
axs[1].plot(x, element_delta_GBB, color="green", label="delta")
axs[1].legend()

# compute DOS by GBB
dos = []
# pick up the min and max value of E_subband
min_temp = []
max_temp = []
for energy in E_subband:
    min_temp.append(energy.get_value()[0])
    max_temp.append(energy.get_value()[nm - 1])
min_subband = min(min_temp).real
max_subband = max(max_temp).real

# define Energy list that should be computed
start = min_subband-0.5
end = max_subband+0.5
step = 0.01
E_list = np.arange(start, end, step)

for energy in E_list:
    GBB_L = surface_gf.surface_gf(energy, eta, Hii[0], Hi1[0].dagger(), Sii[0], TOL=1e-100)[1]
    dos.append(- GBB_L.tre().imag)

# plot energy-band structure
k_total, band = band.band_structure(H, 0.0, 3 * 3.14 / 1.44, 1 * 3.14 / 1.44 / 80)
i = 0
for band_k in band:
    k = np.ones(band[0].get_size()) * k_total[i]
    i += 1
    axs[2].scatter(k, band_k.get_value(), s=10, color="red")
# plot DOS
ax2 = axs[2].twiny()
ax2.plot(dos, E_list, color='green', label="0")
plt.show()