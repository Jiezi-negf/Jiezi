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

file_path_root = "/home/zjy/slurmfile/9649906/"
amountFiles = 9
Vg = []
Id = []
for i in range(1, amountFiles + 1):
    file_path = file_path_root + "process" + str(i) +"/log_print.txt"
    with open(file_path, "r") as f:
        lines = f.readlines()
        ctrlParas = lines[0].split(' ')
        Vg.append(float(ctrlParas[2]))
        lineCurrent = lines[lines.index('ok\n') + 4][10:-2]
        floatLinePhi = list(map(float, lineCurrent.split(',')))
        #the first column(X) to be plot is layerPhi
        Id.append(floatLinePhi[0])
axisVg = np.asarray(Vg).reshape((amountFiles, 1))
axisId = np.asarray(Id).reshape((amountFiles, 1))
axisId = np.absolute(axisId)
fig = plt.figure()
ax = fig.add_subplot(111)
ax2 = ax.twinx()
ax.plot(axisVg, axisId)
ax2.plot(axisVg, np.log10(axisId))
plt.show()

# write the data to file
fileOutput = file_path_root + "IdVgLinear.dat"
dataPlot = np.zeros((amountFiles, 2))
dataPlot[:, 0:1] = axisVg
dataPlot[:, 1:2] = axisId
np.savetxt(fileOutput, dataPlot, fmt='%.18e', delimiter=' ', newline='\n')

fileOutput = file_path_root + "IdVgLog.dat"
dataPlot[:, 1:2] = np.log10(axisId)
np.savetxt(fileOutput, dataPlot, fmt='%.18e', delimiter=' ', newline='\n')
