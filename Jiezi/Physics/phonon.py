# Copyright 2021 Jiezi authors.
#
# This file is part of Jiezi. It is subject to the license terms in the file
# LICENSE.md found in the top-level directory of this distribution. A list of
# Jiezi authors can be found in the file AUTHORS.md at the top-level directory
# of this distribution.
# ==============================================================================


from Jiezi.LA.matrix_numpy import matrix_numpy
from Jiezi.LA import operator as op
from Jiezi.Physics.common import *


def phonon(ee, E_list, form_factor, G_lesser, G_greater, Dac, Dop, omega):
    sigma_lesser_ph = []
    sigma_r_ph = []
    nz = len(form_factor)
    nm = G_lesser[0].get_size[0]
    N_bose1 = bose(E_list[ee + omega] - E_list[ee])
    N_bose2 = bose(E_list[ee] - E_list[ee - omega])
    for zz in range(nz):
        sigma_lesser_ph_zz = matrix_numpy(nm, nm)
        sigma_greater_ph_zz = matrix_numpy(nm, nm)
        sigma_r_ph_zz = matrix_numpy(nm, nm)
        for i in range(nm):
            temp_lesser = 0
            temp_greater = 0
            for j in range(nm):
                # acoustic phonon component
                temp_lesser += Dac * form_factor[zz].get_value(i, j) * \
                               G_lesser[ee][zz].get_value(j, j)
                temp_greater += Dac * form_factor[zz].get_value(i, j) * \
                                G_greater[ee][zz].get_value(j, j)
                # optical phonon component
                temp_lesser += Dop * form_factor[zz].get_value(i, j) * \
                               G_lesser[ee + omega][zz].get_value(j, j) * \
                               (N_bose1 + 1) * heaviside(len(E_list) - (ee + omega))\
                               + Dop * form_factor[zz].get_value(i, j) * \
                               G_lesser[ee - omega][zz].get_value(j, j) * \
                               (N_bose2) * heaviside(ee - omega)
                temp_greater += Dop * form_factor[zz].get_value(i, j) * \
                                G_greater[ee + omega][zz].get_value(j, j) * \
                                (N_bose1) * heaviside(len(E_list) - (ee + omega)) \
                                + Dop * form_factor[zz].get_value(i, j) * \
                                G_greater[ee - omega][zz].get_value(j, j) * \
                                (N_bose2 + 1) * heaviside(ee - omega)
            # avoid numerical issue
            temp_lesser_new = complex(0.0, abs(temp_lesser.imag))
            temp_greater_new = complex(0.0, -abs(temp_greater.imag))
            sigma_lesser_ph_zz.set_value(i, i, temp_lesser_new)
            sigma_greater_ph_zz.set_value(i, i, temp_greater_new)
            # compute retarded phonon self-energy
            sigma_r_ph_zz = op.scamulmat(0.5, op.addmat(sigma_greater_ph_zz, sigma_lesser_ph_zz.nega()))
        sigma_lesser_ph.append(sigma_lesser_ph_zz)
        sigma_r_ph.append(sigma_r_ph_zz)
    return sigma_lesser_ph, sigma_r_ph
