# Copyright 2021 Jiezi authors.
#
# This file is part of Jiezi. It is subject to the license terms in the file
# LICENSE.md found in the top-level directory of this distribution. A list of
# Jiezi authors can be found in the file AUTHORS.md at the top-level directory
# of this distribution.
# ==============================================================================
import sys

from Jiezi.Physics.common import *
from Jiezi.LA import operator as op
import math


def carrierSpectrum(E_list, G_lesser_fullE, G_greater_fullE, volume_cell):
    # number of layer
    num_layer = len(G_lesser_fullE[0])
    # number of energy point
    num_energy = len(E_list)
    # initialize the list to store the spectrum of n and p
    n_spectrum = [None] * num_energy
    p_spectrum = [None] * num_energy

    # for every energy point, compute the carrier on each layer
    for ee in range(num_energy):
        # n/p_spectrum_zz is a list to store the carrier spectrum on zz layer
        n_spectrum_ee = [None] * num_layer
        p_spectrum_ee = [None] * num_layer
        for zz in range(num_layer):
            n_spectrum_ee[zz] = (G_lesser_fullE[ee][zz].real().tre() / volume_cell / math.pi).real
            p_spectrum_ee[zz] = (G_greater_fullE[ee][zz].real().tre() / volume_cell / math.pi).real
        n_spectrum[ee] = n_spectrum_ee
        p_spectrum[ee] = p_spectrum_ee
    return n_spectrum, p_spectrum


def carrierQuantity(E_list, layer_phi_list, n_spectrum, p_spectrum):
    # energy step
    E_step = E_list[1] - E_list[0]
    # number of layer
    num_layer = len(n_spectrum[0])
    # number of energy
    num_energy = len(E_list)
    # initialize the list to store density of carrier on each layer
    n_density = [None] * num_layer
    p_density = [None] * num_layer
    # integrate the n/p spectrum on each layer
    for zz in range(num_layer):
        n_density_zz = 0.0
        p_density_zz = 0.0
        for ee in range(0, num_energy - 1):
            if E_list[ee] > - layer_phi_list[zz]:
                n_density_zz += (n_spectrum[ee][zz] + n_spectrum[ee + 1][zz]) * E_step / 2
            elif E_list[ee] < - layer_phi_list[zz]:
                p_density_zz += (p_spectrum[ee][zz] + p_spectrum[ee + 1][zz]) * E_step / 2
        n_density[zz] = n_density_zz
        p_density[zz] = p_density_zz
    return n_density, p_density


def densityOfStates(E_list, G_R_fullE, volume_cell):
    # number of layer
    num_layer = len(G_R_fullE[0])
    # number of energy point
    num_energy = len(E_list)
    # construct an empty list to store dos
    dos = [None] * num_energy
    # compute density of states(DOS)
    for ee in range(num_energy):
        dos_ee = [None] * num_layer
        for zz in range(num_layer):
            dos_ee[zz] = (- 2 * G_R_fullE[ee][zz].imaginary().tre() / volume_cell / math.pi).real
        dos[ee] = dos_ee
    return dos


def currentSpectrum(E_list, G_lesser_fullE, G_greater_fullE, G1i_lesser_fullE,
                    Sigma_left_lesser_fullE, Sigma_left_greater_fullE,
                    Sigma_right_lesser_fullE, Sigma_right_greater_fullE,
                    Hi1):
    # number of layer
    num_layer = len(G_lesser_fullE[0])
    # number of energy point
    num_energy = len(E_list)
    J_spectrum = [None] * num_energy
    JTimesEnergy_spectrum = [None] * num_energy

    for ee in range(num_energy):
        J_spectrum_ee = [None] * (num_layer + 1)
        JTimesEnergy_spectrum_ee = [None] * (num_layer + 1)
        # compute J_L and JTimesEnergy_L on energy E_list[ee]
        J_spectrum_ee[0] = (op.addmat(op.matmulmat(G_greater_fullE[ee][0],
                                                  Sigma_left_lesser_fullE[ee]),
                                     op.matmulmat(G_lesser_fullE[ee][0],
                                                  Sigma_left_greater_fullE[ee]).nega()).tre()
                             * q_unit / h_bar / math.pi).real
        JTimesEnergy_spectrum_ee[0] = J_spectrum_ee[0] * E_list[ee]
        # compute J(zz->zz+1)
        for zz in range(0, num_layer - 1):
            J_spectrum_ee[zz + 1] = (-2.0 * op.matmulmat(Hi1[zz + 1], G1i_lesser_fullE[ee][zz]).imaginary().tre()
                                    * q_unit / h_bar / math.pi).real
            JTimesEnergy_spectrum_ee[zz + 1] = J_spectrum_ee[zz + 1] * E_list[ee]
        # compute J_R and JTimesEnergy_R on energy E_list[ee]
        J_spectrum_ee[num_layer] = (-1 * op.addmat(op.matmulmat(G_greater_fullE[ee][num_layer - 1],
                                                          Sigma_right_lesser_fullE[ee]),
                                             op.matmulmat(G_lesser_fullE[ee][num_layer - 1],
                                                          Sigma_right_greater_fullE[ee]).nega()).tre()
                                   * q_unit / h_bar / math.pi).real
        JTimesEnergy_spectrum_ee[num_layer] = J_spectrum_ee[num_layer] * E_list[ee]

        J_spectrum[ee] = J_spectrum_ee
        JTimesEnergy_spectrum[ee] = JTimesEnergy_spectrum_ee
    return J_spectrum, JTimesEnergy_spectrum


def currentQuantity(E_list, J_spectrum, JTimesEnergy_spectrum):
    # number of layer
    num_layer = len(J_spectrum[0]) - 1
    # number of energy point
    num_energy = len(E_list)
    # energy step
    E_step = E_list[1] - E_list[0]

    J = [None] * (num_layer + 1)
    JTimesEnergy = [None] * (num_layer + 1)
    for zz in range(num_layer + 1):
        J_zz = 0.0
        JTimesEnergy_zz = 0.0
        for ee in range(num_energy - 1):
            J_zz += (J_spectrum[ee][zz] + J_spectrum[ee + 1][zz]) * E_step / 2.0
            JTimesEnergy_zz += (JTimesEnergy_spectrum[ee][zz] + JTimesEnergy_spectrum[ee + 1][zz]) * E_step / 2.0
        J[zz] = J_zz
        JTimesEnergy[zz] = JTimesEnergy_zz
    return J, JTimesEnergy


def quantity(E_list, G_R_fullE, G_lesser_fullE, G_greater_fullE, G1i_lesser_fullE,
             Sigma_left_lesser_fullE, Sigma_left_greater_fullE,
             Sigma_right_lesser_fullE, Sigma_right_greater_fullE,
             Hi1, volume_cell, U, layer_phi_list, Ec, Eg):
    # number of layer
    num_layer = len(G_lesser_fullE[0])
    # # transfer to real space
    # for zz in range(num_layer):
    #     left, right = op.general_inv(U[zz])
    #     Hi1[zz] = mode2real(Hi1[zz], left, right)
    #     for ee in range(len(E_list)):
    #         G_R_fullE[ee][zz] = mode2real(G_R_fullE[ee][zz], left, right)
    #         G_lesser_fullE[ee][zz] = mode2real(G_lesser_fullE[ee][zz], left, right)
    #         G_greater_fullE[ee][zz] = mode2real(G_greater_fullE[ee][zz], left, right)
    #         if zz == 0:
    #             Sigma_left_lesser_fullE[ee] = mode2real(Sigma_left_lesser_fullE[ee], left, right)
    #             Sigma_left_greater_fullE[ee] = mode2real(Sigma_left_greater_fullE[ee], left, right)
    #         if zz < num_layer - 1:
    #             G1i_lesser_fullE[ee][zz] = mode2real(G1i_lesser_fullE[ee][zz], left, right)
    #         if zz == num_layer - 1:
    #             Sigma_right_lesser_fullE[ee] = mode2real(Sigma_right_lesser_fullE[ee], left, right)
    #             Sigma_right_greater_fullE[ee] = mode2real(Sigma_right_greater_fullE[ee], left, right)

    n_tol = []
    p_tol = []
    J = []
    dos = []
    E_step = E_list[1] - E_list[0]
    # compute J_L
    G_J_L = []
    for ee in range(len(E_list)):
        G_J_L.append(op.addmat(op.matmulmat(G_greater_fullE[ee][0],
                                            Sigma_left_lesser_fullE[ee]),
                               op.matmulmat(G_lesser_fullE[ee][0],
                                            Sigma_left_greater_fullE[ee]).nega()).tre())
    J_L = q_unit * integral(E_list, G_J_L) / math.pi / h_bar
    J.append(J_L.real)

    # compute n, p, and J(i->i+1)
    for i in range(num_layer):
        # # G_n_i[ee]: energy is ee, layer is i. i is constant, ee is different
        # G_n_i = []
        # G_p_i = []
        # # number of sites per layer
        # num_site = G_lesser_fullE[0][i].get_size()[0]

        # average value of phi added on each layer is layer_phi_list[layer_index]
        layer_phi = layer_phi_list[i]
        Ec_layer = Ec - layer_phi
        Ev_layer = Ec - Eg - layer_phi

        result_n = 0.0
        result_p = 0.0

        # # compute n on layer i
        # diff_n = Ec_layer - E_list[0]
        # start_normal = int(diff_n.real // E_step)
        # end_index = len(E_list) - 1
        # if diff_n <= 0:
        #     start_index = 0
        # elif Ec_layer > E_list[end_index]:
        #     start_index = end_index
        # else:
        #     start_index = start_normal
        # for ee in range(start_index, end_index):
        #     result_n += (G_lesser_fullE[ee][i].real().tre() + G_lesser_fullE[ee + 1][i].real().tre()) * E_step / 2
        # result_n = result_n / volume_cell / math.pi
        #
        # # compute p on layer i
        # diff_p = Ev_layer - E_list[0]
        # start_index = 0
        # end_normal = int(diff_p.real // E_step)
        # if diff_p <= 0:
        #     end_index = 0
        # elif Ev_layer > E_list[len(E_list) - 1]:
        #     end_index = len(E_list) - 1
        # else:
        #     end_index = end_normal
        # for ee in range(start_index, end_index):
        #     result_p += (G_greater_fullE[ee][i].real().tre() + G_greater_fullE[ee + 1][i].real().tre()) * E_step / 2
        # result_p = result_p / volume_cell / math.pi
        for ee in range(0, len(E_list) - 1):
            if E_list[ee] > -layer_phi:
                result_n += (G_lesser_fullE[ee][i].real().tre() + G_lesser_fullE[ee + 1][i].real().tre()) * E_step / 2
            elif E_list[ee] < -layer_phi:
                result_p += (G_greater_fullE[ee][i].real().tre() + G_greater_fullE[ee + 1][i].real().tre()) * E_step / 2
        result_n = result_n / volume_cell / math.pi
        result_p = result_p / volume_cell / math.pi



        # # compute the function G(ee) in location i, which will be integrated
        # for ee in range(len(E_list)):
        #     G_n_i.append(G_lesser_fullE[ee][i].real().tre() / volume_cell)
        #     G_p_i.append(G_greater_fullE[ee][i].real().tre() / volume_cell)
        # # n_i, p_i: n of location/layer i
        # n_i = integral(E_list, G_n_i) / math.pi
        # p_i = integral(E_list, G_p_i) / math.pi
        # n_tol.append(n_i)
        # p_tol.append(p_i)
        n_tol.append(result_n)
        p_tol.append(result_p)

    for i in range(0, num_layer - 1):
        # compute J(i->i+1)
        G_J_i = []
        # compute the function G_J_i(ee) in location i, which will be integrated
        for ee in range(len(E_list)):
            G_J_i.append(-2.0 * op.matmulmat(Hi1[i+1], G1i_lesser_fullE[ee][i]).imaginary().tre())

        J_i = q_unit * integral(E_list, G_J_i) / math.pi / h_bar
        J.append(J_i)

    # compute J_R
    G_J_R = []
    for ee in range(len(E_list)):
        G_J_R.append(op.addmat(op.matmulmat(G_greater_fullE[ee][num_layer - 1],
                                            Sigma_right_lesser_fullE[ee]),
                               op.matmulmat(G_lesser_fullE[ee][num_layer - 1],
                                            Sigma_right_greater_fullE[ee]).nega()).tre())
    J_R = - q_unit * integral(E_list, G_J_R) / math.pi / h_bar
    J.append(J_R.real)

    # compute density of states(DOS)
    for ee in range(len(E_list)):
        dos_ee = []
        for zz in range(num_layer):
            dos_ee.append(- 2 * G_R_fullE[ee][zz].imaginary().tre() / volume_cell / math.pi)
        dos.append(dos_ee)

    return n_tol, p_tol, J, dos


def mode2real(E_list, G_R_fullE, G_lesser_fullE, G_greater_fullE, G1i_lesser_fullE,
              Sigma_left_lesser_fullE, Sigma_left_greater_fullE,
              Sigma_right_lesser_fullE, Sigma_right_greater_fullE,
              Hi1, U):
    # number of layer
    num_layer = len(G_lesser_fullE[0])
    # transfer to real space
    for zz in range(num_layer):
        left, right = op.general_inv(U[zz])
        Hi1[zz] = op.trimatmul(left, Hi1[zz], right, "nnn")
        for ee in range(len(E_list)):
            G_R_fullE[ee][zz] = op.trimatmul(left, G_R_fullE[ee][zz], right, "nnn")
            G_lesser_fullE[ee][zz] = op.trimatmul(left, G_lesser_fullE[ee][zz], right, "nnn")
            G_greater_fullE[ee][zz] = op.trimatmul(left, G_greater_fullE[ee][zz], right, "nnn")
            if zz == 0:
                Sigma_left_lesser_fullE[ee] = op.trimatmul(left, Sigma_left_lesser_fullE[ee], right, "nnn")
                Sigma_left_greater_fullE[ee] = op.trimatmul(left, Sigma_left_greater_fullE[ee], right, "nnn")
            if zz < num_layer - 1:
                G1i_lesser_fullE[ee][zz] = op.trimatmul(left, G1i_lesser_fullE[ee][zz], right, "nnn")
            if zz == num_layer - 1:
                Sigma_right_lesser_fullE[ee] = op.trimatmul(left, Sigma_right_lesser_fullE[ee], right, "nnn")
                Sigma_right_greater_fullE[ee] = op.trimatmul(left, Sigma_right_greater_fullE[ee], right, "nnn")
    return G_R_fullE, G_lesser_fullE, G_greater_fullE, G1i_lesser_fullE, \
        Sigma_left_lesser_fullE, Sigma_left_greater_fullE, \
        Sigma_right_lesser_fullE, Sigma_right_greater_fullE
