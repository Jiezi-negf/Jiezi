# Copyright 2021 Jiezi authors.
#
# This file is part of Jiezi. It is subject to the license terms in the file
# LICENSE.md found in the top-level directory of this distribution. A list of
# Jiezi authors can be found in the file AUTHORS.md at the top-level directory
# of this distribution.
# ==============================================================================
import os

from Jiezi.FEM import constant_parameters
from Jiezi.FEM import map
from Jiezi.FEM.mesh import create_dof
from Jiezi.Graph import builder
from Jiezi.Physics import hamilton
from Jiezi.Physics.PrePoisson import PrePoisson
from Jiezi.Physics.band import subband, get_EcEg
from Jiezi.Physics.modespace import mode_space
from Jiezi.Physics.SCBA import SCBA
from Jiezi.Physics.rgf import rgf
from Jiezi.Physics.common import *
from Jiezi.Physics.quantity import quantity
import numpy as np
from Jiezi.Physics.poisson import poisson
import matplotlib.pyplot as plt
from Jiezi.Visualization.toVTK import trans_VTK_xml
import sys
import os

# set the path for writing or reading files
path_Files = os.path.abspath(os.path.join(os.getcwd(), "..", "Files"))

# output the content on console to file
# f = open("print_file.txt", "w+")
# sys.stdout = f

class Logger(object):
    def __init__(self, filename=path_Files + '/log_print.txt', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'w+')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

sys.stdout = Logger(stream=sys.stdout)

# construct the structure
cnt = builder.CNT(n=4, m=0, Trepeat=10, nonideal=False)
cnt.construct()
num_atom_cell = cnt.get_nn()
num_atom_total = cnt.get_nn() * cnt.get_Trepeat()
radius_tube = cnt.get_radius()
length_single_cell = cnt.get_singlecell_length()
z_total = cnt.get_length()
volume_cell = math.pi * radius_tube ** 2 * length_single_cell
print("Amount of atoms in single cell:", num_atom_cell)
print("Total amount of atoms:", num_atom_total)
print("Radius of tube:", radius_tube)
print("Length of single cell:", length_single_cell)
print("Length of whole device:", z_total)
width_cnt_scale = 1
width_oxide_scale = 3
z_length_oxide_scale = 0.3
width_cnt = width_cnt_scale * radius_tube
zlength_oxide = z_length_oxide_scale * z_total
width_oxide = width_oxide_scale * radius_tube
print("Length of gate:", zlength_oxide)
print("Thickness of cnt:", width_cnt)
print("Thickness of oxide:", width_oxide)
r_inter = radius_tube - width_cnt / 2
r_outer = r_inter + width_cnt
r_oxide = r_outer + width_oxide
z_translation = 0.5 * (z_total - zlength_oxide)
geo_para = [r_inter, r_outer, r_oxide, z_total, zlength_oxide, z_translation]
# # use salome to build the FEM grid -- use my own PC
# geo_para, path_xml = PrePoisson(cnt, width_cnt_scale, width_oxide_scale, z_length_oxide_scale)

# read mesh information from .dat file -- use server or other PC
path_dat = path_Files + "/Mesh_whole.dat"

# solve the constant terms and parameters of poisson equation
# path_dat = "/home/zjy/salome/SALOME-9.3.0-UB18.04-SRC/myPro/" + 'Mesh_whole.dat'
info_mesh, dof_amount, dof_coord_list = create_dof(path_dat)
print("amount of element:", len(info_mesh))
print("amount of dof:", dof_amount)

N_GP_T, cell_co, cell_long_term, cell_NJ, cell_NNTJ, mark_list, cnt_cell_list, Dirichlet_list = \
    constant_parameters.constant_parameters(info_mesh, geo_para)
coord_GP_list = constant_parameters.get_coord_GP_list(cell_co)


# cut the whole area to different groups with simple geometry
cut_radius = 3
cut_z = 3
dict_cell = map.cut(r_oxide, z_total, info_mesh, cut_radius, cut_z)


# construct the initial guess of Phi, the value on Dirichlet point is Dirichlet_BC
Dirichlet_BC_source = 1.0
Dirichlet_BC_gate = 1.0
Dirichlet_BC_drain = 1.0
# Dirichlet_BC = np.ones(3) * 1.0
Dirichlet_BC = [Dirichlet_BC_source, Dirichlet_BC_gate, Dirichlet_BC_drain]
phi_guess = np.ones([dof_amount, 1]) * 0.5
for type_i in range(len(Dirichlet_list)):
    for i in range(len(Dirichlet_list[type_i])):
        phi_guess[Dirichlet_list[type_i][i], 0] = Dirichlet_BC[type_i]
print("Dirichlet BC is:", Dirichlet_BC)
print("mu_l:", mul)
print("mu_r:", mur)

# set doping
doping_source = 0
doping_channel = 0
doping_drain = 0
print("doping source, channel, drain are:", doping_source, doping_channel, doping_drain)

# compute the bottom of the conduction band energy Ec and the energy width of the forbidden band Eg(band gap)
# build hamilton matrix
H = hamilton.hamilton(cnt, onsite=-0.28, hopping=-2.97)
H.build_H()
H.build_S(base_overlap=0.018)
Ec, Eg = get_EcEg(H)


while 1:
    # map phi_vec to phi_cell
    phi_cell = map.map_tocell(info_mesh, phi_guess)
    # build hamilton matrix
    H = hamilton.hamilton(cnt, onsite=-0.28, hopping=-2.97)
    H.build_H()
    H.build_S(base_overlap=0.018)
    layer_phi_list = H.H_add_phi(dict_cell, phi_cell, cell_co, cut_radius, cut_z, r_oxide, z_total)

    # compute the eigenvalue(subband energy) and eigenvector(transformation matrix)
    # E_subband is a list, the length of which is equal to number of layers.
    # The length of E_subband[i] is equal to atoms in every single layer.
    # U is a list, the length of which is equal to number of layers.
    # The size of U[i] is equal to every single block matrix of H
    E_subband, U = subband(H, k=0)

    # compute the mode space basis to decrease the size of H
    nm = E_subband[0].get_size()[0]
    nn = nm
    Hii_new, Hi1_new, Sii_new, form_factor, U_new = mode_space(H, U, nn)
    Hii_new = H.get_Hii()
    Hi1_new = H.get_Hi1()
    Sii_new = H.get_Sii()

    # pick up the min and max value of E_subband
    # nm = Hii_new[0].get_size()[0]
    min_temp = []
    max_temp = []
    for energy in E_subband:
        min_temp.append(energy.get_value()[nm//2 - nn//2])
        max_temp.append(energy.get_value()[nm//2 + nn//2 - 1])
    min_subband = min(min_temp).real
    max_subband = max(max_temp).real

    # define Energy list that should be computed
    E_start = min(mul, mur, min_subband) - 10 * KT
    E_end = max(mul, mur, max_subband) + 10 * KT
    E_step = 0.05
    E_list = np.arange(E_start, E_end, E_step)
    print("Energy list:", E_start, "to", E_end, "Step of energy:", E_step)

    # compute GF by RGF iteration
    # define the phonon self-energy matrix as zero matrix
    sigma_r_ph = []
    sigma_lesser_ph = []
    nz = len(Hii_new)
    nm = Hii_new[0].get_size()[0]
    for ee in range(len(E_list)):
        sigma_ph_ee = []
        for j in range(nz):
            sigma_ph_element = matrix_numpy(nm, nm)
            sigma_ph_ee.append(sigma_ph_element)
        sigma_r_ph.append(sigma_ph_ee)
        sigma_lesser_ph.append(sigma_ph_ee)

    # initialize list to store GF matrix of every energy
    iter_max = 10
    TOL = 1e-100
    ratio = 0.5
    eta = 5e-6
    ac = 2.5
    op = 1.0e8
    v_s = 5.0e5
    omega = 4
    # Dac = ac ** 2 * KT / (cnt.get_mass_density() * v_s ** 2)
    # Dop = (h_bar * op) ** 2 / (2 * cnt.get_mass_density() * omega * E_step)
    Dac = 0
    Dop = 0
    G_R_fullE, G_lesser_fullE, G_greater_fullE, G1i_lesser_fullE, \
    Sigma_left_lesser_fullE, Sigma_left_greater_fullE, Sigma_right_lesser_fullE, Sigma_right_greater_fullE = \
        SCBA(E_list, iter_max, TOL, ratio, eta, mul, mur, Hii_new, Hi1_new, Sii_new,
             sigma_lesser_ph, sigma_r_ph, form_factor, Dac, Dop, omega)

    n_tol, p_tol, J, dos = quantity(E_list, G_R_fullE, G_lesser_fullE, G_greater_fullE, G1i_lesser_fullE,
                                    Sigma_left_lesser_fullE, Sigma_left_greater_fullE,
                                    Sigma_right_lesser_fullE, Sigma_right_greater_fullE,
                                    Hi1_new, volume_cell, U_new, layer_phi_list, Ec, Eg)
    print("layer_phi:", layer_phi_list)
    print("n_tol:", n_tol)
    print("p_tol:", p_tol)
    # get dos, n, p and set doping concentration on every Gauss Point of each cell
    dos_GP_list = constant_parameters.get_dos_GP_list(coord_GP_list, dos, z_total, mark_list)
    n_GP_list = constant_parameters.get_np_GP_list(coord_GP_list, n_tol, z_total, mark_list)
    p_GP_list = constant_parameters.get_np_GP_list(coord_GP_list, p_tol, z_total, mark_list)
    # set doping concentration
    doping_GP_list = constant_parameters.doping(coord_GP_list, zlength_oxide, z_translation,
                                                doping_source, doping_drain, doping_channel, mark_list)

    # solve poisson equation
    # set the initial value of ef
    ef_init_n = np.ones([len(info_mesh), 4]) * (-1e2)
    ef_init_p = np.ones([len(info_mesh), 4]) * 1e2
    TOL_ef = 5e-5
    TOL_du = 1e-3
    mode = 2
    phi = poisson(mode, info_mesh, N_GP_T, cell_long_term, cell_NJ, cell_NNTJ, cnt_cell_list,
                  ef_init_n, ef_init_p,
                  Dirichlet_list, Dirichlet_BC, E_list, Ec, Eg, TOL_ef, TOL_du,
                  dos_GP_list, n_GP_list, p_GP_list, doping_GP_list, phi_guess, dof_amount)

    # test if phi is convergence
    TOL_phi = 1e-1
    weight_old = 0.5
    residual = np.linalg.norm(phi - phi_guess, ord=2) / phi.shape[0]
    # residual = math.sqrt(residual.real) / len(phi)
    if residual < TOL_phi:
        # file output
        print("residual between adjacent big loop is:", residual)
        print("ok")
        print("current:", J)
        print("electron:", n_tol)
        print("hole:", p_tol)
        trans_VTK_xml(phi[:, 0].real, dof_coord_list, info_mesh, path_Files)
        # print(dos)
        break
    else:
        phi_guess = phi_guess * weight_old + phi * (1 - weight_old)
        print("residual between adjacent big loop is:", residual)