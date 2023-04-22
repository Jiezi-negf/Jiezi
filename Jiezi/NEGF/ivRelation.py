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

from multiprocessing import Process
from Jiezi.NEGF.serial import serial


if __name__ == "__main__":
    pro_list = []
    Dirichlet_BC_gate = 1.0
    mul = 0.0
    mur_start = -1.0
    mur_step = 0.1
    num_mur = 2
    for i in range(num_mur):
        mur = mur_start + mur_step * i
        p = Process(target=serial, args=(mul, mur, Dirichlet_BC_gate, i))
        p.start()
        pro_list.append(p)
    for i in pro_list:
        i.join()
