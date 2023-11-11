# Copyright 2021 Jiezi authors.
#
# This file is part of Jiezi. It is subject to the license terms in the file
# LICENSE.md found in the top-level directory of this distribution. A list of
# Jiezi authors can be found in the file AUTHORS.md at the top-level directory
# of this distribution.
# ==============================================================================

import numpy as np
from Jiezi.Physics.common import time_it
import warnings
import time


# input parameters
ef = -100000
E_step = 0.005
E_start = -9
E_end = 9
E_list = list(np.arange(E_start, E_end, E_step))
dos = list(np.exp(E_list))


def fermi(x):
    return 1/(1+np.exp(x/0.026))

# numpy method
@ time_it
def quad_np(ef, E_list, E_start, E_step, dos):
    E_list = np.array(E_list)
    dos = np.array(dos)
    zero_index = - int(E_start // E_step)
    E_list_n = E_list[zero_index:]
    # print(E_list[0:zero_index + 1])
    # E_list_p = E_list[0:zero_index + 1]
    # fermi_list = fermi(E_list_n)
    res = np.trapz(fermi(E_list_n - ef) * dos[zero_index:], dx=E_step)
    return res


# iteration method
@ time_it
def quad_iter(ef, E_list, E_start, E_step, dos):
    res = 0
    for ee in range(0, len(E_list) - 1):
        if E_list[ee] < 0:
            continue
        else:
            res += (dos[ee] * fermi(E_list[ee] - ef)
                       + dos[ee + 1] * fermi(E_list[ee + 1] - ef)) * E_step / 2
    return res


print(quad_np(ef, E_list, E_start, E_step, dos))
print(quad_iter(ef, E_list, E_start, E_step, dos))


