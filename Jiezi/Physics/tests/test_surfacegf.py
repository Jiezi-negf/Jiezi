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


cnt = builder.CNT(n=4, m=0, Trepeat=5, nonideal=False)
cnt.construct()
H = hamilton.hamilton(cnt, onsite=-0.28, hopping=-2.97)
H.build_H()
H.build_S(base_overlap=0.018)
E_subband, U = band.subband(H, k=0)
nm = 16
E_list = [0]
Hii = H.get_Hii()
Hi1 = H.get_Hi1()
Sii = H.get_Sii()
nz = len(Hii)
# Hii, Hi1, Sii, form_factor = modespace.mode_space(H, U, "low", nm)
ee = 0
eta = 1e-4
GBB_L = surface_gf.surface_gf(E_list[ee], eta, Hii[0], Hi1[0].dagger(), Sii[0], TOL=1e-12)[1]
GBB_R = surface_gf.surface_gf(E_list[ee], eta, Hii[nz - 1], Hi1[nz], Sii[nz - 1], TOL=1e-12)[1]
delta_GBB = op.addmat(GBB_L, GBB_R.nega())
# print(GBB_L.get_value())
# print(GBB_R.get_value())
# print(np.max(abs(op.addmat(GBB_L, GBB_R.nega()).get_value().real)))
# print(np.sum(abs(op.addmat(GBB_L, GBB_R.nega()).get_value().real)))
G00_L_ref = surface_gf.surface_gf(E_list[ee], eta, Hii[0], Hi1[0].dagger(), Sii[0], TOL=1e-12)[0]
G00_L_dumb = surface_gf.surface_gf_dumb(E_list[ee], eta, Hii[nz - 1], Hi1[nz].dagger(), Sii[nz - 1], TOL=1e-12)
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
plt.subplot(1, 2, 1)
plt.title("sancho-rubio vs dumb")
plt.plot(x, element_clever, color="red", label="sancho-rubio")
plt.plot(x, element_dumb, color="blue", label="dumb")
plt.plot(x, element_delta, color="green", label="delta")
plt.legend()
plt.subplot(1, 2, 2)
plt.title("GBB from left and right by sancho-rubio method")
plt.plot(x, element_GBB_L, color="red", label="GBB from left")
plt.plot(x, element_GBB_R, color="blue", label="GBB from right")
plt.plot(x, element_delta_GBB, color="green", label="delta")
plt.legend()
plt.show()