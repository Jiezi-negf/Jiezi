
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
from Jiezi.Physics.common import *
import numpy as np
from Jiezi.Physics import surface_gf
from Jiezi.Physics import hamilton, band
from Jiezi.Graph import builder


nz = 6
super_factor = 2
cnt = builder.CNT(n=16, m=0, Trepeat=nz, nonideal=False)
cnt.construct()
phi = 0.0
H = hamilton.hamilton(cnt, onsite=-phi, hopping=-2.97)
H.build_H()
H.build_S(base_overlap=0.00)


Hii_cell = H.get_Hii()[1]
Hi1_cell = H.get_Hi1()[1]
Sii_cell = H.get_Sii()[1]
Si1_cell = H.get_Si1()[1]
Hii, Hi1 = hamilton.H_extendSize(Hii_cell, Hi1_cell, super_factor)
size_old = Hii_cell.get_size()[0]
size_new = int(size_old * super_factor)
Sii = matrix_numpy(size_new, size_new, "float")
for i in range(super_factor):
    Sii.set_block_value(i * size_old, (i + 1) * size_old, i * size_old, (i + 1) * size_old,
                            Sii_cell.get_value())

num_E = 1000
E_list = np.linspace(-6, 6, num_E)
eta_sg = 5e-6

nm = Hii_cell.get_size()[0]
transmission_list = np.zeros((num_E, 2))
E_listX = E_list.reshape((num_E, 1))
transmission_list[0:, 0:1] = E_listX



nm = int(nm * super_factor)
nz = int(nz / super_factor)
for ee in range(num_E):
    w = complex(E_list[ee], 0.0)
    # compute the surface GF of left lead
    G00_L = surface_gf.surface_gf(E_list[ee], eta_sg, Hii, Hi1.dagger(), Sii, TOL=1e-10)[0]
    # compute the self energy of left lead based on the surface GF
    Sigma_L = op.trimatmul(Hi1, G00_L, Hi1, type="cnn")
    Gamma_L = op.scamulmat(complex(0.0, 1.0),
                           op.addmat(Sigma_L, Sigma_L.dagger().nega()))

    # compute the surface GF of right lead
    G00_R = surface_gf.surface_gf(E_list[ee], eta_sg, Hii, Hi1, Sii, TOL=1e-10)[0]
    # compute the self energy of right lead based on the surface GF
    Sigma_R = op.trimatmul(Hi1, G00_R, Hi1, type="nnc")
    Gamma_R = op.scamulmat(complex(0.0, 1.0),
                           op.addmat(Sigma_R, Sigma_R.dagger().nega()))

    # construct the whole Hamiltonian matrix and the Sigma matrix
    H_total = matrix_numpy(nz * nm, nz * nm)
    for i in range(nz):
        H_total.set_block_value(i * nm, (i + 1) * nm, i * nm, (i + 1) * nm, Hii)
        if i > 0:
            H_total.set_block_value((i - 1) * nm, i * nm, i * nm, (i + 1) * nm, Hi1)
            H_total.set_block_value(i * nm, (i + 1) * nm, (i - 1) * nm, i * nm, Hi1.dagger())

    # construct the whole overlap matrix
    S_total = matrix_numpy(nz * nm, nz * nm)
    for i in range(nz):
        S_total.set_block_value(i * nm, (i + 1) * nm, i * nm, (i + 1) * nm, Sii)

    # construct the whole Sigma matrix and Sigma_lesser matrix
    Sigma_total = matrix_numpy(nz * nm, nz * nm)
    Sigma_total.set_block_value(0, nm, 0, nm, Sigma_L)
    Sigma_total.set_block_value((nz - 1) * nm, nz * nm, (nz - 1) * nm, nz * nm, Sigma_R)

    Gamma_L_total = matrix_numpy(nz * nm, nz * nm)
    Gamma_R_total = matrix_numpy(nz * nm, nz * nm)
    Gamma_L_total.set_block_value(0, nm, 0, nm, Gamma_L)
    Gamma_R_total.set_block_value((nz - 1) * nm, nz * nm, (nz - 1) * nm, nz * nm, Gamma_R)

    # compute the GF of the whole system directly
    G_R_inv = op.inv(op.addmat(op.scamulmat(w, S_total), H_total.nega(), Sigma_total.nega()))

    # only reserve the diagonal block, set the other blocks to be zero
    G_R_inv_total = matrix_numpy(nz * nm, nz * nm)
    for i in range(nz):
        G_R_inv_total.set_block_value(i * nm, (i + 1) * nm, i * nm, (i + 1) * nm,
                                      G_R_inv.get_value(i * nm, (i + 1) * nm, i * nm, (i + 1) * nm))

    # compute current of coherent transport by transmission method
    transmission = op.matmulmat(op.trimatmul(Gamma_L_total, G_R_inv, Gamma_R_total, type="nnn"),
                                G_R_inv.dagger()).tre().real

    transmission_list[ee, 1] = transmission

path_Files = os.path.abspath(os.path.join(__file__, "../..", "Files"))
fileName = "/transmitsuper1000.dat"
fname = path_Files + fileName
np.savetxt(fname, transmission_list, fmt='%.18e', delimiter=' ', newline='\n')
