import os
import sys
sys.path.append(os.path.abspath(__file__ + "/../../../.."))
from Jiezi.Physics.band import subband, band2dos
from Jiezi.Physics.rgf_origin import rgf
from Jiezi.Physics.common import *
import numpy as np
from Jiezi.FEM.ef_solver import brent
from Jiezi.Physics import quantity
import matplotlib.pyplot as plt


from Jiezi.Physics import hamilton
from Jiezi.Graph import builder


nz = 30
cnt = builder.CNT(n=19, m=0, Trepeat=nz, nonideal=False)
cnt.construct()
L = cnt.get_singlecell_length()
H = hamilton.hamilton(cnt, onsite=0, hopping=-2.7)
H.build_H()
H.build_S(base_overlap=0.018)
Hii_0 = H.get_Hii()[1]
Hi1_0 = H.get_Hi1()[1]


phi = 0
mul = 0.0
mur = 0.0
H = hamilton.hamilton(cnt, onsite=-phi-0, hopping=-2.97)
H.build_H()
H.build_S(base_overlap=0.018)
layer_phi_list = [phi] * nz
# volume_cell = 263.98
volume_cell = np.pi * cnt.get_radius() ** 2 * L

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
#
# # pick up the min and max value of E_subband
# min_temp = []
# max_temp = []
# for energy in E_subband:
#     min_temp.append(energy.get_value()[0])
#     max_temp.append(energy.get_value()[nm - 1])
# min_subband = min(min_temp).real
# max_subband = max(max_temp).real
#
#
# start = -1 + 0.0005
# end = 1
# step = 0.01
# E_list = np.arange(start, end, step)
# print("E_start:", start, "E_end:", end)
#
# sigma_ph = []
# for i in range(len(E_list)):
#     sigma_ph_ee = []
#     for j in range(nz):
#         sigma_ph_element = matrix_numpy(nm, nm)
#         sigma_ph_ee.append(sigma_ph_element)
#     sigma_ph.append(sigma_ph_ee)
#
# G_lesser_fullE = [None] * len(E_list)
# G_greater_fullE = [None] * len(E_list)
# for ee in range(len(E_list)):
#     w = complex(E_list[ee], eta_gf)
#     # this is the output of rgf method
#     G_R_rgf, G_lesser, G_greater, G1i_lesser, Sigma_left_lesser, Sigma_left_greater, \
#         Sigma_right_lesser, Sigma_right_greater = \
#         rgf(ee, E_list, eta_sg, mul, mur, Hii, Hi1, Sii, sigma_ph, sigma_ph)
#     G_lesser_fullE[ee] = G_lesser
#     G_greater_fullE[ee] = G_greater
# n_spectrum_rgf, p_spectrum_rgf = quantity.carrierSpectrum(E_list, G_lesser_fullE, G_greater_fullE, volume_cell)
# n_tol_rgf, p_tol_rgf = quantity.carrierQuantity(E_list, layer_phi_list, n_spectrum_rgf, p_spectrum_rgf)
# print(n_tol_rgf)
n = 131 * L / volume_cell * 1e-8
print("n is:", n)
E_start_poi = -1
E_end_poi = 3
E_step_poi = 0.01
E_list_poi = np.arange(E_start_poi, E_end_poi, E_step_poi)

# set k points and compute band
dense = 1000
k_points = np.arange(-np.pi / L, np.pi / L, 2*np.pi / (nz * L) / dense)
bandArray = hamilton.compute_band(Hii_0, Hi1_0, L, k_points)
# set energy domain of interest
dos0 = band2dos(bandArray, E_list_poi)
dos = list(dos0[:, 1] / (dense * nz * E_step_poi) / volume_cell)

zero_index = - int(E_start_poi // E_step_poi)
E_list_n = E_list_poi[zero_index:]
E_list_p = E_list_poi[0:zero_index + 1]
TOL_ef = 1e-4

n = 131 * L / volume_cell * 1e-8
print("n is:", n)
ef = brent("n", zero_index, E_list_poi, E_list_n, E_step_poi, 0, 0, phi,
                           dos, n,
                           E_list_poi[0]-10, E_list_poi[len(E_list_poi) - 1]+10, 0, TOL_ef)
print("ef is:", ef)

# ef_set = [ef + phi]
# for ef_i in ef_set:
#     result = np.trapz(2 * np.asarray(dos)[zero_index:] * fermi(E_list_n - ef_i), dx=E_step_poi)
#     print(ef, result)
# print(ef)
#
# ef_p = brent("p", zero_index, E_list, E_list_p, E_step_poi, 0, 0, phi,
#                            dos, 1e-22,
#                            E_list[0]-10, E_list[len(E_list) - 1]+10, 0, TOL_ef)
# print(ef_p)
