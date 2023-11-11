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
from Jiezi.Physics import hamilton
from Jiezi.Graph import builder

cnt = builder.CNT(n=8, m=0, Trepeat=3, nonideal=False)
cnt.construct()
H = hamilton.hamilton(cnt, onsite=-0.28, hopping=-2.97)
H.build_H()
H.build_S(base_overlap=0.018)
Hii = H.get_Hii()
Hi1 = H.get_Hi1()
lead_H00_L, lead_H00_R = H.get_lead_H00()
lead_H10_L, lead_H10_R = H.get_lead_H10()
print("Tight binding hamiltonian matrix Hii of (8, 0) CNT is:", Hii[1].get_value())
print("Tight binding hamiltonian matrix Hi1 of (8, 0) CNT is:", Hi1[1].get_value())