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
from Jiezi.Physics.band import subband, band2dos
from Jiezi.Physics.rgf_origin import rgf
from Jiezi.Physics.common import *
import numpy as np
import matplotlib.pyplot as plt


from Jiezi.Physics import hamilton
from Jiezi.Graph import builder


nz = 10
cnt = builder.CNT(n=5, m=5, Trepeat=nz, nonideal=False)
cnt.construct()
phi = 0.0
mul = 0.0
mur = 0.0
H = hamilton.hamilton(cnt, onsite=-phi-0, hopping=-2.97)
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
eta_gf = 5e-6
nm = Hii[0].get_size()[0]

# pick up the min and max value of E_subband
min_temp = []
max_temp = []
for energy in E_subband:
    min_temp.append(energy.get_value()[0])
    max_temp.append(energy.get_value()[nm - 1])
min_subband = min(min_temp).real
max_subband = max(max_temp).real


start = -3 - 0.0001
end = 3
step = 0.001
E_list = np.arange(start, end, step)
print("E_start:", start, "E_end:", end)


dos_rgf = np.zeros((nz, len(E_list)))
sigma_ph = []
for i in range(len(E_list)):
    sigma_ph_ee = []
    for j in range(nz):
        sigma_ph_element = matrix_numpy(nm, nm)
        sigma_ph_ee.append(sigma_ph_element)
    sigma_ph.append(sigma_ph_ee)

for ee in range(len(E_list)):
    w = complex(E_list[ee], eta_gf)
    # this is the output of rgf method
    G_R_rgf, G_lesser, G_greater, G1i_lesser, Sigma_left_lesser, Sigma_left_greater, \
        Sigma_right_lesser, Sigma_right_greater = \
        rgf(ee, E_list, eta_sg, mul, mur, Hii, Hi1, Sii, sigma_ph, sigma_ph)
    dos_rgf[1, ee] = - 1 * G_R_rgf[1].tre().imag / np.pi

# # <<<<<<<<<<<<<<<<part: band2dos<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# Hii_z = H.get_Hii()[1]
# Hi1_z = H.get_Hi1()[1]
# L = cnt.get_singlecell_length()
# k_points = np.arange(-np.pi / L, np.pi / L, 2*np.pi/nz/L/1000)
# bandArray = hamilton.compute_band(Hii_z, Hi1_z, L, k_points)
# dos0 = band2dos(bandArray, E_list)
# dos = list(dos0[:, 1] / 1000 / step / nz)
#
# plt.plot(E_list, dos_rgf[1, :], label="dos_rgf")
# plt.plot(E_list, dos, label="dosFromBand")
# plt.legend()
# plt.show()

# write it to file
path = os.path.abspath(__file__ + "/../../Files")
fileName = "/dos55.dat"
num_energy = len(E_list)
lines = [None] * num_energy
for ee in range(num_energy):
    line = str(E_list[ee]) + '\t' + str(dos_rgf[1, ee]) + '\n'
    lines[ee] = line
with open(path + fileName, 'w') as f:
    f.writelines(lines)



