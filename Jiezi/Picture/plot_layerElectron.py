# Copyright 2021 Jiezi authors.
#
# This file is part of Jiezi. It is subject to the license terms in the file
# LICENSE.md found in the top-level directory of this distribution. A list of
# Jiezi authors can be found in the file AUTHORS.md at the top-level directory
# of this distribution.
# ==============================================================================


import sys
import os

import numpy as np

sys.path.append(os.path.abspath(__file__ + "/../../.."))
import matplotlib.pyplot as plt

file_path_root = "/home/zjy/slurmfile/gra/"
# read axis-Z
nz = 60
amountFiles = 9
dataEZ = np.loadtxt(file_path_root + "process9/SpectrumXYForOthers.dat")
axisZ = dataEZ[0:nz, 1:2]
dataPlot = np.zeros((nz, amountFiles + 1))
# the first column(X) to be plot is the axis Z
dataPlot[:, 0:1] = axisZ
# # read axisE
# axisE = dataEZ[:, 0].take(np.arange(0, dataEZ.shape[0], nz))
for i in range(1, amountFiles + 1):
    file_path = file_path_root + "process" + str(i) +"/log_print.txt"
    with open(file_path, "r") as f:
        lines = f.readlines()
        lineElectron = lines[lines.index('ok\n') + 6][11:-2]
        floatLineElectron = list(map(float, lineElectron.split(',')))
        #the first column(X) to be plot is layerPhi
        dataPlot[:, i:i+1] = np.array(floatLineElectron).reshape((nz, 1))
        plt.plot(axisZ, floatLineElectron, label=str(i))


# write the data to file
fileOutput = file_path_root + "layerElectron.dat"
np.savetxt(fileOutput, dataPlot, fmt='%.18e', delimiter=' ', newline='\n')
plt.legend()
plt.show()

