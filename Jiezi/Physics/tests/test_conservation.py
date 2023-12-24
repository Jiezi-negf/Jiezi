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
# E_start = -9.710972730767262 + 6
# nz = 51
# file_path = "/home/zjy/slurmfile/slurm-8775385-8780746/normal/process6/currentSpectrum.dat"

E_start = -1.66
nz = 60
E_step = 0.001
file_path = "/home/zjy/slurmfile/9717404/process7/currentSpectrum.dat"
# file_path = "/home/zjy/slurmfile/slurm-9576981/process1/electronSpectrum.dat"

with open(file_path, "r") as f:
    lines = f.readlines()
length_lines = len(lines)
num_energyPoints = int(length_lines/nz)
JspectrumWhole = []
JEspectrumWhole = []
E_list = []
for ee in range(num_energyPoints):
    E_list.append(E_start + E_step * ee)
for i in range(nz):
    Jspectrum = [float] * num_energyPoints
    JEspectrum = [float] * num_energyPoints
    for ee in range(num_energyPoints):
        Jspectrum[ee] = float(lines[nz * ee + i].split()[0])
        JEspectrum[ee] = Jspectrum[ee] * (E_start + E_step * ee)
    JspectrumWhole.append(Jspectrum)
    JEspectrumWhole.append(JEspectrum)
plt.subplot(2, 2, 1)
plt.title("J_0")
plt.plot(E_list, JspectrumWhole[0])
plt.subplot(2, 2, 3)
plt.title("JE_0")
plt.plot(E_list, JEspectrumWhole[0])
plt.subplot(2, 2, 2)
plt.title("J_14")
plt.plot(E_list, JspectrumWhole[20])
plt.subplot(2, 2, 4)
plt.title("JE_14")
plt.plot(E_list, JEspectrumWhole[20])
plt.show()



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