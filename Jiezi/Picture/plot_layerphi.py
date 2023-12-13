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
nz = 30
amountFiles = 9
dataEZ = np.loadtxt(file_path_root + "process9/" + "SpectrumXYForOthers.dat")
axisZ = dataEZ[0:nz, 1:2]
dataPlot = np.zeros((nz, amountFiles + 1))
EcPlot = np.zeros((nz, amountFiles + 1))
EvPlot = np.zeros((nz, amountFiles + 1))
# the first column(X) to be plot is the axis Z
dataPlot[:, 0:1] = axisZ
EcPlot[:, 0:1] = axisZ
EvPlot[:, 0:1] = axisZ
Ec = 0
Ev = 0
# # read axisE
# axisE = dataEZ[:, 0].take(np.arange(0, dataEZ.shape[0], nz))
for i in range(9, amountFiles + 1):
    file_path = file_path_root + "process" + str(i) +"/log_print.txt"
    with open(file_path, "r") as f:
        lines = f.readlines()
        for j in range(len(lines)):
            line = lines[j]
            if "Ec" in lines[j]:
                Ec = float(lines[j].split(' ')[2][0:-1])
                Ev = float(lines[j + 1].split(' ')[2][0:-1])
                break
        linePhi = lines[lines.index('ok\n') + 1][12:-2]
        floatLinePhi = list(map(float, linePhi.split(',')))
        #the first column(X) to be plot is layerPhi
        dataPlot[:, i:i+1] = np.array(floatLinePhi).reshape((nz, 1))
        # plt.plot(axisZ, floatLinePhi, label=str(i))
EcPlot[:, 1:] = np.ones((nz, amountFiles)) * Ec - dataPlot[:, 1:]
EvPlot[:, 1:] = np.ones((nz, amountFiles)) * Ev - dataPlot[:, 1:]
for i in range(1, amountFiles + 1):
    plt.plot(axisZ, EcPlot[:, i:i + 1], label="Ec" + str(i))
    plt.plot(axisZ, EvPlot[:, i:i + 1], label="Ev" + str(i))
# write the data to file
fileOutput = file_path_root + "layerPhi.dat"
np.savetxt(fileOutput, dataPlot, fmt='%.18e', delimiter=' ', newline='\n')
fileOutput = file_path_root + "layerEc.dat"
np.savetxt(fileOutput, EcPlot, fmt='%.18e', delimiter=' ', newline='\n')
fileOutput = file_path_root + "layerEv.dat"
np.savetxt(fileOutput, EvPlot, fmt='%.18e', delimiter=' ', newline='\n')
plt.legend()
plt.show()

