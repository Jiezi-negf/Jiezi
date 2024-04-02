# Copyright 2021 Jiezi authors.
#
# This file is part of Jiezi. It is subject to the license terms in the file
# LICENSE.md found in the top-level directory of this distribution. A list of
# Jiezi authors can be found in the file AUTHORS.md at the top-level directory
# of this distribution.
# ==============================================================================


from Jiezi.Physics.phonon import phonon
from Jiezi.Physics.rgf_origin import rgf
from Jiezi.LA import operator as op
from Jiezi.Physics.common import time_it
import time
import numpy as np
import copy


@ time_it
def SCBAopen(E_list, iter_max: int, TOL, ratio, eta, mul, mur, Hii, Hi1, Sii, S00,
            lead_H00_L, lead_H00_R, lead_H10_L, lead_H10_R,
            sigma_lesser_ph, sigma_r_ph, form_factor, Dac, Dop, energyOP):
    print("NEGF solver(SCBA loop) start")
    # initialize
    iter_c = 0
    nz = len(sigma_lesser_ph[0])
    nm = sigma_lesser_ph[0][0].get_size()[0]

    G_R_fullE = [None] * len(E_list)
    G_lesser_fullE = [None] * len(E_list)
    G_greater_fullE = [None] * len(E_list)
    G1i_lesser_fullE = [None] * len(E_list)

    sigma_lesser_ph_fullE = copy.deepcopy(sigma_lesser_ph)
    sigma_r_ph_fullE = copy.deepcopy(sigma_r_ph)
    sigma_lesser_ph_fullE_new = copy.deepcopy(sigma_lesser_ph)
    sigma_r_ph_fullE_new = copy.deepcopy(sigma_r_ph)

    Sigma_left_lesser_fullE = [None] * len(E_list)
    Sigma_left_greater_fullE = [None] * len(E_list)
    Sigma_right_lesser_fullE = [None] * len(E_list)
    Sigma_right_greater_fullE = [None] * len(E_list)

    while iter_c <= iter_max:
        # phonon result ---> GF
        time0 = time.time()
        for ee in range(len(E_list)):
            G_R_ee, G_lesser_ee, G_greater_ee, G1i_lesser_ee, \
            Sigma_left_lesser_ee, Sigma_left_greater_ee, Sigma_right_lesser_ee, Sigma_right_greater_ee = \
                rgf(ee, E_list, eta, mul, mur, Hii, Hi1, Sii, 
                    sigma_lesser_ph_fullE, sigma_r_ph_fullE)
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
            Sigma_right_lesser_fullE[ee] = Sigma_right_lesser_ee
            Sigma_right_greater_fullE[ee] = Sigma_right_greater_ee
        time1 = time.time()
        print("time cost by SCBA rgf part:", time1 - time0)
        # GF result ---> phonon
        time0 = time.time()
        for ee in range(len(E_list)):
            sigma_lesser_ph_ee, sigma_r_ph_ee = \
                phonon(ee, E_list, form_factor, G_lesser_fullE, G_greater_fullE, Dac, Dop, energyOP)
            sigma_lesser_ph_fullE_new[ee] = sigma_lesser_ph_ee
            sigma_r_ph_fullE_new[ee] = sigma_r_ph_ee
        time1 = time.time()
        print("time cost by SCBA phonon part:", time1 - time0)
        # evaluate error
        error = 0.0
        for ee in range(len(E_list)):
            for zz in range(nz):
                error = error + np.sum(np.absolute(sigma_lesser_ph_fullE[ee][zz].get_value()
                                    - sigma_lesser_ph_fullE_new[ee][zz].get_value())) 
        error = error/(len(E_list) * nz * nm * nm)

        # store the error
        if error < TOL:
            print("error between GF and Phonon is:", error)
            print("SCBA convergence reached successfully with "+str(iter_c)+" iterations")
            break
        else:
            # renew the sigma_lesser_ph_fullE, sigma_r_ph_fullE_new
            print("error between GF and Phonon is:", error)
            for ee in range(len(E_list)):
                for zz in range(nz):
                    sigma_lesser_ph_fullE[ee][zz] = op.addmat(op.scamulmat(ratio, sigma_lesser_ph_fullE[ee][zz]),
                                                              op.scamulmat(1 - ratio, sigma_lesser_ph_fullE_new[ee][zz])
                                                              )
                    sigma_r_ph_fullE[ee][zz] = op.addmat(op.scamulmat(ratio, sigma_r_ph_fullE[ee][zz]),
                                                         op.scamulmat(1 - ratio, sigma_r_ph_fullE_new[ee][zz])
                                                         )
        iter_c += 1
    return G_R_fullE, G_lesser_fullE, G_greater_fullE, G1i_lesser_fullE, \
           Sigma_left_lesser_fullE, Sigma_left_greater_fullE, Sigma_right_lesser_fullE, Sigma_right_greater_fullE

