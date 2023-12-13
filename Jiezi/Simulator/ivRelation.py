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

from multiprocessing import Pool
from Jiezi.Simulator.normal import normal

if __name__ == "__main__":
    input_list = []
    mul = 0.0
    mur = -0.3
    V_gate_start = -1.0
    V_gate_step = 0.0
    num_iter = 4
    for i in range(num_iter):
        V_gate = V_gate_start + i * V_gate_step
        input_list.append((mul, mur, V_gate, i))
    pool = Pool(processes=num_iter)
    pool.starmap_async(normal, input_list)
    pool.close()
    pool.join()



