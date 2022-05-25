# Copyright 2021 Jiezi authors.
#
# This file is part of Jiezi. It is subject to the license terms in the file
# LICENSE.md found in the top-level directory of this distribution. A list of
# Jiezi authors can be found in the file AUTHORS.md at the top-level directory
# of this distribution.
# ==============================================================================
from Jiezi.LA.vector_numpy import vector_numpy
from Jiezi.Physics.common import *
from Jiezi.Graph import builder
from Jiezi.Physics import hamilton


def algorithm(chirality_n, chirality_m, T_repeat, a_cc, onsite, hopping, nonideal, base_overlap):
    cnt = builder.CNT(chirality_n, chirality_m, T_repeat, a_cc=a_cc, onsite=onsite, hopping=hopping,
                      nonideal=nonideal)
    cnt.construct()
    H_cell = cnt.get_hamilton_cell()
    H_hopping = cnt.get_hamilton_hopping()
    nn = H_cell.shape[0]
    hopping_value = cnt.get_hopping_value()
    H = hamilton.hamilton(H_cell, H_hopping, nn, T_repeat)
    H.build_H()
    H.build_S(hopping_value, base_overlap=base_overlap)
    Hii = H.get_Hii()
    Hi1 = H.get_Hi1()
    Sii = H.get_Sii()
