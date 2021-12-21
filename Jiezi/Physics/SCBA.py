# Copyright 2021 Jiezi authors.
#
# This file is part of Jiezi. It is subject to the license terms in the file
# LICENSE.md found in the top-level directory of this distribution. A list of
# Jiezi authors can be found in the file AUTHORS.md at the top-level directory
# of this distribution.
# ==============================================================================


from Jiezi.Physics.phonon import phonon
from Jiezi.Physics.rgf import rgf
from Jiezi.LA import operator as op


def SCBA(E_list, iter_max: int, TOL, ratio, eta, mul, mur, Hii, Hi1, Sii, sigma_lesser_ph, sigma_r_ph,
         form_factor, Dac, Dop, omega):
    G_R_fullE = []
    G_lesser_fullE = []
    G_greater_fullE = []
    G1i_lesser_fullE = []
    sigma_lesser_ph_fullE = []
    sigma_r_ph_fullE = []
    Sigma_left_lesser_fullE = []
    Sigma_left_greater_fullE = []
    sigma_lesser_ph_fullE_new = []
    sigma_r_ph_fullE_new = []
    # initialize
    for ee in range(len(E_list)):
        G_R_fullE.append(sigma_lesser_ph[0])
        G_lesser_fullE.append(sigma_lesser_ph[0])
        G_greater_fullE.append(sigma_lesser_ph[0])
        G1i_lesser_fullE.append(sigma_lesser_ph[0])
        sigma_lesser_ph_fullE.append(sigma_lesser_ph[ee])
        sigma_r_ph_fullE.append(sigma_r_ph[ee])
        Sigma_left_lesser_fullE.append(sigma_lesser_ph[0][0])
        Sigma_left_greater_fullE.append(sigma_lesser_ph[0][0])
        sigma_lesser_ph_fullE_new.append(sigma_lesser_ph[0])
        sigma_r_ph_fullE_new.append(sigma_lesser_ph[0])
    iter = 0
    error = 0
    nz = len(sigma_lesser_ph[0])
    nm = len(sigma_lesser_ph[0][0])
    while iter <= iter_max and error <= TOL:
        iter += 1
        # phonon result ---> GF
        for ee in range(len(E_list)):
            G_R_ee, G_lesser_ee, G_greater_ee, G1i_lesser_ee, Sigma_left_lesser_ee, Sigma_left_greater_ee = \
                rgf(ee, eta, mul, mur, Hii, Hi1, Sii, sigma_lesser_ph_fullE, sigma_r_ph_fullE)
            # G_R_fullE, G_lesser_fullE, G_greater_fullE, G1i_lesser_fullE : [[], [], ...]
            # for example, length of G_lesser_fullE is len(E_list)
            # length of G_lesser_fullE[ee] is nz
            # G_lesser_fullE[ee][zz] is a matrix_numpy(nm, nm) object
            G_R_fullE[ee] = G_R_ee
            G_lesser_fullE[ee] = G_lesser_ee
            G_greater_fullE[ee] = G_greater_ee
            G1i_lesser_fullE[ee] = G1i_lesser_ee
            # Sigma_left_lesser_fullE, Sigma_left_greater_fullE: []
            # for example, length of Sigma_left_lesser_fullE is len(E_list)
            # Sigma_left_lesser_fullE[ee] is a matrix_numpy(nm, nm) object
            Sigma_left_lesser_fullE[ee] = Sigma_left_lesser_ee
            Sigma_left_greater_fullE[ee] = Sigma_left_greater_ee

        # GF result ---> phonon
        for ee in range(len(E_list)):
            sigma_lesser_ph_ee, sigma_r_ph_ee = \
                phonon(ee, form_factor, G_lesser_fullE, G_greater_fullE, Dac, Dop, omega)
            sigma_lesser_ph_fullE_new[ee] = sigma_lesser_ph_ee
            sigma_r_ph_fullE_new[ee] = sigma_r_ph_ee

        # evaluate error
        # renew the sigma_lesser_ph_fullE, sigma_r_ph_fullE_new
        error = 0
        for ee in range(len(E_list)):
            for zz in range(nz):
                for n in range(nm):
                    error = error + abs(abs(sigma_lesser_ph_fullE[ee][zz].get_value(n, n))
                                        -abs(sigma_lesser_ph_fullE_new[ee][zz].get_value(n, n)))
                sigma_lesser_ph_fullE[ee][zz] = op.addmat(op.scamulmat(ratio, sigma_lesser_ph_fullE[ee][zz]),
                                                          op.scamulmat(1 - ratio, sigma_lesser_ph_fullE_new[ee][zz]))
                sigma_r_ph_fullE[ee][zz] = op.addmat(op.scamulmat(ratio, sigma_r_ph_fullE[ee][zz]),
                                                     op.scamulmat(1 - ratio, sigma_r_ph_fullE_new[ee][zz]))
        error = error/(len(E_list) * nz * nm)
        print("iter number is:", iter)
        print("error is:", error)
    return G_R_fullE, G_lesser_fullE, G_greater_fullE, G1i_lesser_fullE, Sigma_left_lesser_fullE, \
           Sigma_left_greater_fullE
