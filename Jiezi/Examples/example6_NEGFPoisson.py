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

from Jiezi.FEM import constant_parameters, map
from Jiezi.FEM.mesh import create_dof
from Jiezi.Graph import builder
from Jiezi.Physics import hamilton, quantity
from Jiezi.Physics.band import subband, get_EcEg
from Jiezi.Physics.modespace import mode_space
from Jiezi.Physics.SCBA import SCBA
from Jiezi.Physics.common import *
import numpy as np
from Jiezi.Physics.poisson import poisson
from Jiezi.Visualization.Data2File import phi2VTK, spectrumXY2Dat, spectrumZ2Dat


@time_it
def NEGFPoisson(mul, mur, Dirichlet_BC_gate, weight_old, tol_loop, process_id):
    # <<<<<<<<<<<<<set the path for Input or output files<<<<<<<<<<<<<<<<<<<<<<<
    # path_Files is the path of the shared files among process
    path_Files = os.path.abspath(os.path.join(__file__, "../..", "Files"))
    # path_process_Files is the private files of each process
    path_process_Files = os.path.abspath(os.path.join(path_Files, "normal", "process" + str(process_id)))
    print(path_process_Files)
    # make the file folder
    folder = os.path.exists(path_process_Files)
    if not folder:
        os.makedirs(path_process_Files)

    # <<<<<<<<<<<<<<<redirect the IO-stream to both console and file<<<<<<<<<<<<<<<<<
    class Logger(object):
        def __init__(self, filename):
            self.console = sys.stdout
            self.file = open(filename, 'w+')

        def write(self, message):
            self.console.write(message)
            self.file.write(message)
            self.flush()

        def flush(self):
            self.console.flush()
            self.file.flush()
    sys.stdout = Logger(path_process_Files + '/log_print.txt')
    print(mul, mur, Dirichlet_BC_gate, weight_old, tol_loop, process_id)


    # <<<<<<<<<<<<<<<<<construct the structure<<<<<<<<<<<<<<<<<<<<<<<<<
    cnt = builder.CNT(n=8, m=0, Trepeat=60, nonideal=False)
    cnt.construct()
    num_atom_cell = cnt.get_nn()
    num_cell = cnt.get_Trepeat()
    num_atom_total = num_atom_cell * num_cell
    radius_tube = cnt.get_radius()
    length_single_cell = cnt.get_singlecell_length()
    z_total = cnt.get_length()
    num_supercell = 1
    volume_cell = math.pi * radius_tube ** 2 * length_single_cell
    print("Amount of atoms in single cell:", num_atom_cell)
    print("Total amount of atoms:", num_atom_total)
    print("Radius of tube:", radius_tube)
    print("Length of single cell:", length_single_cell)
    print("Length of whole device:", z_total)
    width_cnt_scale = 1
    width_oxide_scale = 3.18
    z_length_oxide_scale = 0.167
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
    z_isolation = 10
    geo_para = [r_inter, r_outer, r_oxide, z_total, zlength_oxide, z_translation, z_isolation]

    # <<<<<<<<<<<<<<<<<<<read mesh information from .dat file<<<<<<<<<<<<<<<<<<<<<
    path_dat = path_Files + "/Mesh_whole.dat"
    info_mesh, dof_amount, dof_coord_list = create_dof(path_dat)
    print("amount of element:", len(info_mesh))
    print("amount of dof:", dof_amount)

    # <<<<<<<<<<<<<<<<<<<compute the invariant quantity in newton iteration in FEM<<<<<<<<<<<<<
    N_GP_T, cell_co, cell_long_term, cell_NJ, cell_NNTJ, mark_list, cnt_cell_list, Dirichlet_list = \
        constant_parameters.constant_parameters(info_mesh, geo_para)
    coord_GP_list = constant_parameters.get_coord_GP_list(cell_co)


    # <<<<<<<<<<<<<<<<<<<<<<<<<<<cut region for FEM<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # cut the whole area to different groups with simple geometry in order to speed up the process of assigning
    # any point to its corresponding cell
    cut_radius = 3
    cut_z = 3
    dict_cell = map.cut(r_oxide, z_total, info_mesh, cut_radius, cut_z)

    # <<<<<<<<<<<<<<<<<<<<<<<<<set initial guess and boundary conditions<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # construct the initial guess of Phi, the value on Dirichlet point is Dirichlet_BC
    Dirichlet_BC_source = 1.0
    Dirichlet_BC_drain = 1.0
    Dirichlet_BC = [Dirichlet_BC_source, Dirichlet_BC_gate, Dirichlet_BC_drain]
    phi_guess = np.ones([dof_amount, 1]) * 0.5 * (Dirichlet_BC_gate + Dirichlet_BC_source)
    for type_i in range(len(Dirichlet_list)):
        for i in range(len(Dirichlet_list[type_i])):
            phi_guess[Dirichlet_list[type_i][i], 0] = Dirichlet_BC[type_i]
    print("Dirichlet BC is:", Dirichlet_BC)
    print("source electrostatic doping on z axis is from", 0, "to", z_translation - z_isolation)
    print("drain electrostatic doping on z axis is from",
          z_translation + zlength_oxide + z_isolation, "to", z_total)
    print("mu_l:", mul)
    print("mu_r:", mur)

    # <<<<<<<<<<<<<<<<<<<set doping, fixed charge and point-defect<<<<<<<<<<<<<<<<<<<<<<<<<<
    # set doping
    doping_source = 0
    doping_channel = 0
    doping_drain = 0
    print("doping source, channel, drain are:", doping_source, doping_channel, doping_drain)
    # set fixed charge parameters in oxide
    fixedChargeDensity = 0
    fixedChargeScope = [r_outer, (r_outer + r_oxide) / 2, 0, z_total]
    # construct doping concentration
    doping_GP_list = constant_parameters.doping(coord_GP_list, zlength_oxide, z_translation,
                                                doping_source, doping_drain, doping_channel, mark_list)
    # construct fixed charge in oxide
    fixedCharge_GP_list = constant_parameters.fixedChargeInit(coord_GP_list)
    constant_parameters.addFixedCharge(fixedCharge_GP_list, coord_GP_list, mark_list,
                                       fixedChargeScope, fixedChargeDensity)
    print("total fixed charge in oxide--scope:\n",
          "radius from", fixedChargeScope[0], "to", fixedChargeScope[1], '\n',
          "coordinate on z axis from", fixedChargeScope[2], "to", fixedChargeScope[3], '\n',
          "fixed charge in oxide--density:", fixedChargeDensity)


    # <<<<<<<<<<<<<<<<<<<<<compute Ec and Eg<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # compute the bottom of the conduction band energy Ec and the energy width of the forbidden band Eg(band gap)
    # build hamilton matrix
    H = hamilton.hamilton(cnt, onsite=-0.28, hopping=-2.97)
    H.build_H()
    H.build_S(base_overlap=0.018)
    Ec, Eg = get_EcEg(H)

    # <<<<<<<<<<<<<<<<<<<<<NEGF-Poisson Loop<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # set iter_max_big to control the big loop
    iter_big_max = 25
    iter_big = 0
    while iter_big < iter_big_max:
        iter_big += 1
        # display which loop is running
        print("\n")
        print("Loop", iter_big)

        # <<<<<<<<<<<<<<<<<map phi_vec to phi_cell<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        phi_cell = map.map_tocell(info_mesh, phi_guess)


        # <<<<<<<<<<<<<<<<<build hamilton matrix with phi<<<<<<<<<<<<<<<<<<<<<<<<<<
        H = hamilton.hamilton(cnt, onsite=-0.28, hopping=-2.97)
        H.build_H()
        H.build_S(base_overlap=0.018)
        layer_phi_list = H.H_add_phi(dict_cell, phi_cell, cell_co, cut_radius, cut_z, r_oxide, z_total, num_supercell)

        # <<<<<<<<<<<<<<<<<<<<Set the energy points to be computed<<<<<<<<<<<<<<<<<<<<<
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

        # pick up the min and max value of E_subband
        # nm = Hii_new[0].get_size()[0]
        min_temp = []
        max_temp = []
        for energy in E_subband:
            min_temp.append(energy.get_value()[nm // 2 - nn // 2])
            max_temp.append(energy.get_value()[nm // 2 + nn // 2 - 1])
        min_subband = min(min_temp).real
        max_subband = max(max_temp).real

        # define Energy list that should be computed
        E_start = min(mul, mur, min_subband) - 10 * KT + 9
        E_end = max(mul, mur, max_subband) + 10 * KT - 7
        E_step = 0.001
        E_list = np.arange(E_start, E_end, E_step)
        print("Energy list:", E_start, "to", E_end, "Step of energy:", E_step)

        # <<<<<<<<<<<<<<<<<get useful Hamiltonian matrix, define phonon self-energy<<<<<<<<<<<<<<<<<<
        # get all the quantities which will be used in RGF from H object
        Hii_new = H.get_Hii()
        Hi1_new = H.get_Hi1()
        Sii_new = H.get_Sii()
        lead_H00_L, lead_H00_R = H.get_lead_H00()
        lead_H10_L, lead_H10_R = H.get_lead_H10()
        S00 = H.get_S00()
        # initialize the phonon self-energy matrix as zero matrix
        sigma_r_ph = []
        sigma_lesser_ph = []
        nz = num_cell
        nm = Hii_new[1].get_size()[0]
        for ee in range(len(E_list)):
            sigma_ph_ee = []
            for j in range(nz):
                sigma_ph_element = matrix_numpy(nm, nm)
                sigma_ph_ee.append(sigma_ph_element)
            sigma_r_ph.append(sigma_ph_ee)
            sigma_lesser_ph.append(sigma_ph_ee)

        # define parameters used in SCBA loop
        iter_SCBA_max = 10
        TOL_SCBA = 1e-100
        ratio_SCBA = 0.5
        eta = 5e-6
        ac = 2.5
        op = 1.0e8
        v_s = 5.0e5
        omega = 4
        # Dac = ac ** 2 * KT / (cnt.get_mass_density() * v_s ** 2)
        # Dop = (h_bar * op) ** 2 / (2 * cnt.get_mass_density() * omega * E_step)
        Dac = 0
        Dop = 0

        # <<<<<<<<<<<<<<<<<<SCBA Loop<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        G_R_fullE, G_lesser_fullE, G_greater_fullE, G1i_lesser_fullE, \
            Sigma_left_lesser_fullE, Sigma_left_greater_fullE, Sigma_right_lesser_fullE, Sigma_right_greater_fullE = \
            SCBA(E_list, iter_SCBA_max, TOL_SCBA, ratio_SCBA, eta, mul, mur, Hii_new, Hi1_new, Sii_new, S00,
                 lead_H00_L, lead_H00_R, lead_H10_L, lead_H10_R,
                 sigma_lesser_ph, sigma_r_ph, form_factor, Dac, Dop, omega)

        # <<<<<<<<<<<<<<<<<<<<<physical quantity from NEGF<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        n_spectrum, p_spectrum = quantity.carrierSpectrum(E_list, G_lesser_fullE, G_greater_fullE, volume_cell)
        n_tol, p_tol = quantity.carrierQuantity(E_list, layer_phi_list, n_spectrum, p_spectrum)
        dos = quantity.densityOfStates(E_list, G_R_fullE, volume_cell)

        print("layer_phi:", layer_phi_list)
        print("n_tol:", n_tol)
        print("p_tol:", p_tol)
        # get dos, n, p and set doping concentration on every Gauss Point of each cell
        dos_GP_list = constant_parameters.get_dos_GP_list(coord_GP_list, dos, z_total, mark_list)
        n_GP_list = constant_parameters.get_np_GP_list(coord_GP_list, n_tol, z_total, mark_list)
        p_GP_list = constant_parameters.get_np_GP_list(coord_GP_list, p_tol, z_total, mark_list)

        # <<<<<<<<<<<<<<<<<<<<<<solve poisson equation<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        # set the initial value of ef
        ef_init_n = np.ones([len(info_mesh), 4]) * (-1e2)
        ef_init_p = np.ones([len(info_mesh), 4]) * 1e2
        TOL_ef = 1e-4
        TOL_du = 1e-4
        iter_NonLinearPoisson_max = 20
        mode = 1
        phi = poisson(mode, info_mesh, N_GP_T, cell_long_term, cell_NJ, cell_NNTJ, cnt_cell_list,
                      ef_init_n, ef_init_p, mark_list,
                      Dirichlet_list, Dirichlet_BC, E_list, Ec, Eg, TOL_ef, TOL_du, iter_NonLinearPoisson_max,
                      dos_GP_list, n_GP_list, p_GP_list, doping_GP_list, fixedCharge_GP_list, phi_guess, dof_amount)

        # test if phi is convergence
        residual = np.linalg.norm(phi - phi_guess, ord=2) / phi.shape[0]
        # residual = math.sqrt(residual.real) / len(phi)
        if residual < tol_loop:
            print("residual between adjacent big loop is:", residual)
            print("ok")
            phi_cell = map.map_tocell(info_mesh, phi)
            layer_phi_list = H.H_add_phi(dict_cell, phi_cell, cell_co, cut_radius, cut_z, r_oxide, z_total,
                                         num_supercell)
            print("layer_phi:", layer_phi_list)
            # compute the final current energy spectrum and density which is a function of position and energy
            J_spectrum, JTimesEnergy_spectrum = quantity.currentSpectrum(
                E_list, G_lesser_fullE, G_greater_fullE, G1i_lesser_fullE,
                Sigma_left_lesser_fullE, Sigma_left_greater_fullE,
                Sigma_right_lesser_fullE, Sigma_right_greater_fullE,
                Hi1_new)
            J, JTimesEnergy = quantity.currentQuantity(E_list, J_spectrum, JTimesEnergy_spectrum)
            # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<data output<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            print("current:", J)
            print("current times energy:", JTimesEnergy)
            print("electron:", n_tol)
            print("hole:", p_tol)
            phi2VTK(phi[:, 0].real, dof_coord_list, info_mesh, path_process_Files)
            spectrumXY2Dat(E_list, length_single_cell, num_cell, num_supercell,
                           path_process_Files, "SpectrumXYForCurrent.dat")
            spectrumXY2Dat(E_list, length_single_cell, num_cell, num_supercell,
                           path_process_Files, "SpectrumXYForOthers.dat")
            spectrumZ2Dat(J_spectrum, path_process_Files, "currentSpectrum.dat")
            spectrumZ2Dat(dos, path_process_Files, "densityOfState.dat")
            spectrumZ2Dat(n_spectrum, path_process_Files, "electronSpectrum.dat")
            spectrumZ2Dat(p_spectrum, path_process_Files, "holeSpectrum.dat")
            print(dos)
            break
        else:
            print("phi norm:", np.linalg.norm(phi, ord=2))
            phi_guess = phi_guess * weight_old + phi * (1 - weight_old)
            print("residual between adjacent big loop is:", residual)
    if iter_big == iter_big_max:
        print("big loop reach iteration times limit:", iter_big_max)
    return


if __name__ == "__main__":
    mul = float(sys.argv[1])
    mur = float(sys.argv[2])
    V_gate = float(sys.argv[3])
    weight_old = float(sys.argv[4])
    tol_loop = float(sys.argv[5])
    process_id = int(sys.argv[6])
    NEGFPoisson(mul, mur, V_gate, weight_old, tol_loop, process_id)
NEGFPoisson(0, -0.4, -0.8, 0.5, 1e-4, 0)