# Copyright 2021 Jiezi authors.
#
# This file is part of Jiezi. It is subject to the license terms in the file
# LICENSE.md found in the top-level directory of this distribution. A list of
# Jiezi authors can be found in the file AUTHORS.md at the top-level directory
# of this distribution.
# ==============================================================================
from Jiezi.Physics.hamilton import hamilton
from Jiezi.LA import operator as op
from Jiezi.LA.matrix_numpy import matrix_numpy
from Jiezi.Physics.surface_gf import surface_gf
from Jiezi.Physics.common import *


def rgf(ee, eta, mul, mur, Hii, Hi1, Sii, sigma_lesser_ph, sigma_r_ph):
    g_R = []
    g_lesser = []
    G_R = []
    G_lesser = []
    G_greater = []
    G1i_lesser = []
    nz = len(Hii)
    # 1 compute left contact's surface GF and self-energy
    H00 = op.addmat(Hii[0], op.matmulmat(sigma_r_ph[ee][0], Sii[0]))
    G00 = surface_gf(E_list[ee], eta, H00, Hi1[0].dagger(), Sii[0], iter_max=50, TOL=1)
    Sigma_left_R = op.trimatmul(Hi1[0], G00, Hi1[0], type="cnn")
    Sigma_left_lesser = op.scamulmat(-fermi(E_list[ee]-mul), op.addmat(Sigma_left_R, Sigma_left_R.dagger().nega()))
    Sigma_left_greater = op.scamulmat(-(1-fermi(E_list[ee]-mul)),
                                      op.addmat(Sigma_left_R, Sigma_left_R.dagger().nega()))
    # 2 the first member of little retarded GF -- g_0_R
    w = complex(E_list[ee], eta)
    g_0_R = op.inv(op.addmat(op.scamulmat(w, Sii[0]), Hii[0].nega(), Sigma_left_R.nega()))
    g_R.append(g_0_R)
    # 3 the first member of little lesser GF -- g_0_lesser
    g_0_lesser = op.trimatmul(g_0_R, op.addmat(sigma_lesser_ph[ee][0], Sigma_left_lesser), g_0_R, type="nnc")
    g_lesser.append(g_0_lesser)
    # 4 derive little retarded GF from i=1 to i=nz-2  -- g_i_R
    for i in range(1, nz-1):
        g_i_R = op.inv(op.addmat(op.scamulmat(w, Sii[i]), Hii[i].nega(), sigma_r_ph[ee][i].nega(),
                                 op.trimatmul(Hi1[i], g_R[i - 1], Hi1[i], type="cnn").nega()))
        g_R.append(g_i_R)
    # 5 derive little lesser GF from i=1 to i=nz-2  -- g_i_lesser
    for i in range(1, nz-1):
        g_i_lesser = op.trimatmul(g_R[i], op.addmat(sigma_lesser_ph[ee][i],
                                    op.trimatmul(Hi1[i], g_lesser[i-1], Hi1[i], type="cnn")), g_R[i], type="nnc")
        g_lesser.append(g_i_lesser)
    # 6 compute right contact's surface GF and self-energy
    H00 = op.addmat(Hii[nz-1], op.matmulmat(sigma_r_ph[ee][nz-1], Sii[nz-1]))
    G00 = surface_gf(E_list[ee], eta, H00, Hi1[nz], Sii[nz-1], iter_max=50, TOL=1)
    Sigma_right_R = op.trimatmul(Hi1[nz], G00, Hi1[nz], type="nnc")
    Sigma_right_lesser = op.scamulmat(-fermi(E_list[ee]-mur), op.addmat(Sigma_right_R, Sigma_right_R.dagger().nega()))
    # 7 compute the last member of real retarded GF -- G_R[nz-1]
    # for convenience, first fill all members of the G_R list
    temp = matrix_numpy()
    for i in range(nz):
        G_R.append(temp)
    G_R[nz-1] = op.inv(op.addmat(op.scamulmat(w, Sii[nz-1]), Hii[nz-1].nega(), sigma_r_ph[ee][nz-1].nega(),
                                 op.trimatmul(Hi1[nz-1], g_lesser[nz-2], Hi1[nz-1], type="cnn").nega(),
                                 Sigma_right_R.nega()))
    # 8 compute the last member of real lesser GF -- G_lesser[nz-1]
    # for convenience, first fill all members of the G_lesser list
    for i in range(nz):
        G_lesser.append(temp)
    G_lesser[nz-1] = op.trimatmul(G_R[nz-1], op.addmat(sigma_lesser_ph[ee][nz-1],
                                  op.trimatmul(Hi1[nz-1], g_R[nz-2], Hi1[nz-1], type="cnn"),
                                  Sigma_right_lesser), G_R[nz-1], type="nnc")
    # 9.1 derive real retarded GF from i=nz-2 to i=0  -- G_i_R
    for i in range(nz-2, -1, -1):
        G_R[i] = op.addmat(g_R[i], op.trimatmul(g_R[i], op.trimatmul(Hi1[i+1], G_R[i+1], Hi1[i+1], type="nnc"),
                                                g_R[i], type="nnn"))
    # 9.2 derive real lesser GF from i=nz-2 to i=0  -- G_i_lesser
    for i in range(nz-2, -1, -1):
        G_lesser[i] = op.addmat(g_lesser[i],
             op.trimatmul(g_lesser[i], op.trimatmul(Hi1[i+1], G_R[i+1], Hi1[i+1], type="ncc"), g_R[i], type="nnc"),
             op.trimatmul(g_R[i], op.trimatmul(Hi1[i+1], G_lesser[i+1], Hi1[i+1], type="nnc"), g_R[i], type="nnc"),
             op.trimatmul(g_R[i], op.trimatmul(Hi1[i+1], G_R[i+1], Hi1[i+1], type="nnc"), g_lesser[i], type="nnn"))
    # 9.3 derive real lesser GF from i=0 to i=nz-2  -- G1i_lesser
    for i in range(0, nz-1):
        G1i_lesser_i = op.addmat(op.trimatmul(G_R[i+1], Hi1[i+1], g_lesser[i], type="ncn"),
                                 op.trimatmul(G_lesser[i+1], Hi1[i+1], g_R[i], type="ncc"))
        G1i_lesser.append(G1i_lesser_i)
    # 9.4 derive real greater GF from i=0 to i=nz-1  -- G_i_greater
    for i in range(0, nz):
        G_i_greater = op.addmat(G_lesser[i], G_R[i], G_R[i].dagger().nega())
        G_greater.append(G_i_greater)
    return G_R, G_lesser, G_greater, G1i_lesser, Sigma_left_lesser, Sigma_left_greater
