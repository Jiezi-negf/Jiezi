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


def phonon(ee, E_list, form_factor, G_lesser, G_greater, Dac, Dop, energyOP):
    E_step = E_list[1] - E_list[0]
    nz = len(form_factor)
    nm = G_lesser[0][0].get_size()[0]
    sigma_lesser_ph = [None] * nz
    sigma_r_ph = [None] * nz
    for zz in range(nz):
        sigma_lesser_ph_zz = matrix_numpy(nm, nm)
        sigma_greater_ph_zz = matrix_numpy(nm, nm)
        # acoustic phonon component
        sigma_lesser_ph_zz = op.addmat(sigma_lesser_ph_zz, op.scamulmat(Dac, G_lesser[ee][zz]))
        sigma_greater_ph_zz = op.addmat(sigma_greater_ph_zz, op.scamulmat(Dac, G_greater[ee][zz]))
        # optical phonon component
        for k in range(len(energyOP)):
            omega = int(energyOP[k] / E_step)
            if len(E_list) > (ee + omega):
                N_bose1 = bose(E_list[ee + omega] - E_list[ee])
                sigma_lesser_ph_zz = op.addmat(sigma_lesser_ph_zz,
                                               op.scamulmat(Dop[k] * (N_bose1 + 1) *
                                                            heaviside(len(E_list) - (ee + omega)),
                                                            G_lesser[ee + omega][zz]))
                sigma_greater_ph_zz = op.addmat(sigma_greater_ph_zz,
                                                op.scamulmat(Dop[k] * (N_bose1) *
                                                             heaviside(len(E_list) - (ee + omega)),
                                                             G_greater[ee + omega][zz]))
            if ee > omega:
                N_bose2 = bose(E_list[ee] - E_list[ee - omega])
                sigma_lesser_ph_zz = op.addmat(sigma_lesser_ph_zz,
                                               op.scamulmat(Dop[k] * (N_bose2) *
                                                            heaviside(ee - omega),
                                                            G_lesser[ee - omega][zz]))
                sigma_greater_ph_zz = op.addmat(sigma_greater_ph_zz,
                                                op.scamulmat(Dop[k] * (N_bose2 + 1) *
                                                             heaviside(ee - omega),
                                                             G_greater[ee - omega][zz]))
        # avoid numerical issue
        # sigma_lesser_ph_zz.copy(np.absolute(sigma_lesser_ph_zz.real().get_value()))
        # sigma_greater_ph_zz.copy(-1 * np.absolute(sigma_greater_ph_zz.real().get_value()))
        # compute retarded phonon self-energy
        sigma_r_ph_zz = op.scamulmat(complex(0, -0.5), op.addmat(sigma_lesser_ph_zz, sigma_greater_ph_zz))
        sigma_lesser_ph[zz] = sigma_lesser_ph_zz.diag()
        sigma_r_ph[zz] = sigma_r_ph_zz.diag()
    return sigma_lesser_ph, sigma_r_ph

def phononPara(cnt):
    n, m = cnt.get_chirality()
    atomNum = cnt.get_nn()
    diameter_nm = cnt.get_radius() * 2 / 10
    dac = 9
    mu_s = 2.11e4
    Dac = dac ** 2 * KB_J * T / (cnt.get_mass() * mu_s ** 2)
    energyIntraLO = 0.19
    energyIntraRBM = float(format(0.028 / diameter_nm, '.3f'))
    energyInterLOTA = 0.18
    energyOP = [energyIntraLO, energyIntraRBM, energyInterLOTA]
<<<<<<< HEAD
    DopIntraLO = 9.8e-3 * 16 / n * 8
    DopIntraRBM = 0.54e-3 * 16 / n * 8
    DopInterLOTA = 19.3e-3 * 16 / n * 8
    Dop = [DopIntraLO, DopIntraRBM, DopInterLOTA]
    return energyOP, Dop, Dac
=======
    DopIntraLO = 9.8e-3 * 16 / n
    DopIntraRBM = 0.54e-3 * 16 / n
    DopInterLOTA = 19.3e-3 * 16 / n
    Dop = [DopIntraLO, DopIntraRBM, DopInterLOTA]
    return energyOP, Dop, Dac
>>>>>>> 4a04ebaf72dc14a26e08205ee24ded2a97f1b2cc
