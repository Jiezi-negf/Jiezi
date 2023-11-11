# Copyright 2021 Jiezi authors.
#
# This file is part of Jiezi. It is subject to the license terms in the file
# LICENSE.md found in the top-level directory of this distribution. A list of
# Jiezi authors can be found in the file AUTHORS.md at the top-level directory
# of this distribution.
# ==============================================================================


import os
import sys
import numpy as np
import unittest

sys.path.append(os.path.abspath(__file__ + "/../../.."))
from Jiezi.Physics import hamilton, surface_gf, rgf
from Jiezi.Graph import builder
from Jiezi.Physics.common import *
from Jiezi.LA import operator as op
from Jiezi.LA.matrix_numpy import matrix_numpy


class TestBandStructure(unittest.TestCase):

    def test_surfaceGF(self):
        cnt = builder.CNT(n=5, m=0, Trepeat=3, nonideal=False)
        cnt.construct()
        H = hamilton.hamilton(cnt, onsite=-0.28, hopping=-2.97)
        H.build_H()
        H.build_S(base_overlap=0.018)
        Hii = H.get_Hii()
        Hi1 = H.get_Hi1()
        Sii = H.get_Sii()
        energy = 0
        eta = 1e-2
        G00_L = surface_gf.surface_gf(energy, eta, Hii[0], Hi1[0].dagger(), Sii[0], TOL=1e-12)[0]
        G00_L_dumb = surface_gf.surface_gf_dumb(energy, eta, Hii[0], Hi1[0].dagger(), Sii[0], TOL=1e-12)
        np.testing.assert_array_almost_equal(G00_L_dumb.get_value(), G00_L.get_value())

    def test_rgf(self):
        cnt = builder.CNT(n=6, m=3, Trepeat=6, nonideal=False)
        cnt.construct()
        H = hamilton.hamilton(cnt, onsite=-0.28, hopping=-2.97)
        H.build_H()
        H.build_S(base_overlap=0.018)
        Hii = H.get_Hii()
        Hi1 = H.get_Hi1()
        Sii = H.get_Sii()
        S00 = H.get_S00()
        lead_H00_L, lead_H00_R = H.get_lead_H00()
        lead_H10_L, lead_H10_R = H.get_lead_H10()
        E_list = [-0.3]
        ee = 0
        eta_rgf = 5e-6
        eta_sg = 5e-6
        w = complex(E_list[ee], 0.0)
        sigma_ph = []
        nz = len(Hii)
        nm = Hii[0].get_size()[0]
        for i in range(len(E_list)):
            sigma_ph_ee = []
            for j in range(nz):
                sigma_ph_element = matrix_numpy(nm, nm)
                sigma_ph_ee.append(sigma_ph_element)
            sigma_ph.append(sigma_ph_ee)
        mul = 0
        mur = 0
        # this is the output of rgf method
        G_R_rgf, G_lesser, G_greater, G1i_lesser, Sigma_left_lesser, Sigma_left_greater, \
            Sigma_right_lesser, Sigma_right_greater = \
            rgf.rgf(ee, E_list, eta_rgf, mul, mur, Hii, Hi1, Sii, S00,
                        lead_H00_L, lead_H00_R, lead_H10_L, lead_H10_R,
                        sigma_ph, sigma_ph)

        # compute the surface GF of left lead
        G00_L = surface_gf.surface_gf(E_list[ee], eta_sg, Hii[0], Hi1[0].dagger(), Sii[0], TOL=1e-10)[0]
        # compute the self energy of left lead based on the surface GF
        Sigma_L = op.trimatmul(Hi1[0], G00_L, Hi1[0], type="cnn")
        Gamma_L = op.scamulmat(complex(0.0, 1.0),
                               op.addmat(Sigma_L, Sigma_L.dagger().nega()))
        Sigma_lesser_L = op.scamulmat(fermi(E_list[ee] - mul), Gamma_L)
        Sigma_greater_L = op.scamulmat(1.0 - fermi(E_list[ee] - mul), Gamma_L)

        # compute the surface GF of right lead
        G00_R = surface_gf.surface_gf(E_list[ee], eta_sg, Hii[nz - 1], Hi1[nz], Sii[nz - 1], TOL=1e-10)[0]
        # compute the self energy of right lead based on the surface GF
        Sigma_R = op.trimatmul(Hi1[nz], G00_R, Hi1[nz], type="nnc")
        Gamma_R = op.scamulmat(complex(0.0, 1.0),
                               op.addmat(Sigma_R, Sigma_R.dagger().nega()))
        Sigma_lesser_R = op.scamulmat(fermi(E_list[ee] - mur), Gamma_R)
        Sigma_greater_R = op.scamulmat(1.0 - fermi(E_list[ee] - mur), Gamma_R)

        # construct the whole Hamiltonian matrix and the Sigma matrix
        H_total = matrix_numpy(nz * nm, nz * nm)
        for i in range(nz):
            H_total.set_block_value(i * nm, (i + 1) * nm, i * nm, (i + 1) * nm, Hii[i])
            if i > 0:
                H_total.set_block_value((i - 1) * nm, i * nm, i * nm, (i + 1) * nm, Hi1[i])
                H_total.set_block_value(i * nm, (i + 1) * nm, (i - 1) * nm, i * nm, Hi1[i].dagger())

        # construct the whole overlap matrix
        S_total = matrix_numpy(nz * nm, nz * nm)
        for i in range(nz):
            S_total.set_block_value(i * nm, (i + 1) * nm, i * nm, (i + 1) * nm, Sii[i])
            # if i > 0:
            #     S_total.set_block_value((i - 1) * nm, i * nm, i * nm, (i + 1) * nm, Si1[i])
            #     S_total.set_block_value(i * nm, (i + 1) * nm, (i - 1) * nm, i * nm, Si1[i].dagger())

        # construct the whole Sigma matrix and Sigma_lesser matrix
        Sigma_total = matrix_numpy(nz * nm, nz * nm)
        Sigma_total.set_block_value(0, nm, 0, nm, Sigma_L)
        Sigma_total.set_block_value((nz - 1) * nm, nz * nm, (nz - 1) * nm, nz * nm, Sigma_R)

        Gamma_L_total = matrix_numpy(nz * nm, nz * nm)
        Gamma_R_total = matrix_numpy(nz * nm, nz * nm)
        Gamma_L_total.set_block_value(0, nm, 0, nm, Gamma_L)
        Gamma_R_total.set_block_value((nz - 1) * nm, nz * nm, (nz - 1) * nm, nz * nm, Gamma_R)

        Sigma_lesser_total = matrix_numpy(nz * nm, nz * nm)
        Sigma_lesser_total.set_block_value(0, nm, 0, nm, Sigma_lesser_L)
        Sigma_lesser_total.set_block_value((nz - 1) * nm, nz * nm, (nz - 1) * nm, nz * nm, Sigma_lesser_R)

        Sigma_greater_total = matrix_numpy(nz * nm, nz * nm)
        Sigma_greater_total.set_block_value(0, nm, 0, nm, Sigma_greater_L)
        Sigma_greater_total.set_block_value((nz - 1) * nm, nz * nm, (nz - 1) * nm, nz * nm, Sigma_greater_R)

        # compute the GF of the whole system directly
        G_R_inv = op.inv(op.addmat(op.scamulmat(w, S_total), H_total.nega(), Sigma_total.nega()))
        G_lesser_inv = op.trimatmul(G_R_inv, Sigma_lesser_total, G_R_inv, type="nnc")
        G_greater_inv = op.addmat(G_lesser_inv.nega(),
                                  op.scamulmat(complex(0.0, 1.0),
                                               op.addmat(G_R_inv, G_R_inv.dagger().nega())))

        # only reserve the diagonal block, set the other blocks to be zero
        G_R_inv_total = matrix_numpy(nz * nm, nz * nm)
        for i in range(nz):
            G_R_inv_total.set_block_value(i * nm, (i + 1) * nm, i * nm, (i + 1) * nm,
                                          G_R_inv.get_value(i * nm, (i + 1) * nm, i * nm, (i + 1) * nm))

        # transfer the block matrix G_R_rgf to the whole matrix G_R_rgf_total
        G_R_rgf_total = matrix_numpy(nz * nm, nz * nm)
        for i in range(nz):
            G_R_rgf_total.set_block_value(i * nm, (i + 1) * nm, i * nm, (i + 1) * nm, G_R_rgf[i])

        np.testing.assert_array_almost_equal(G_R_rgf_total.get_value(), G_R_inv_total.get_value())

if __name__ == "__main__":
    unittest.main()