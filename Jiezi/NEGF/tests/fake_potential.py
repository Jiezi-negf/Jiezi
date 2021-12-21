# Copyright 2021 Jiezi authors.
#
# This file is part of Jiezi. It is subject to the license terms in the file
# LICENSE.md found in the top-level directory of this distribution. A list of
# Jiezi authors can be found in the file AUTHORS.md at the top-level directory
# of this distribution.
# ==============================================================================
import numpy as np


def fake_potential(T_repeat, E_bottom, E_top):
    phi = np.zeros(T_repeat)
    phi[0] = E_bottom
    for i in range(1, T_repeat // 2):
        phi[i] = (E_top - E_bottom)/(T_repeat/2 - 2) * (i - 1) + E_bottom
    for i in range(T_repeat // 2, T_repeat):
        phi[i] = phi[T_repeat - 1 - i]
    return phi
