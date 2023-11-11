import matplotlib.pyplot as plt
import numpy as np
import time
from Jiezi.LA.matrix_numpy import matrix_numpy
from Jiezi.LA import operator as op

# num = 500
# time0 = time.time()
# a = [np.exp(i) for i in np.arange(num)]
# time1 = time.time()
# print("Time cost by np.arange with loop comprehension:", time1 - time0)
# time0 = time.time()
# b = np.exp(np.arange(num))
# time1 = time.time()
# print("Time cost by np.arange with vectorized operation:", time1 - time0)
# time0 = time.time()
# c = [np.exp(i) for i in range(num)]
# time1 = time.time()
# print("Time cost by range:", time1 - time0)
#
# d = [None] * num
# time0 = time.time()
# for i in range(num):
#     d[i] = np.exp(i)
# time1 = time.time()
# print("Time cost by range without loop comprehension:", time1 - time0)
#
#
# e = [None] * num
# time0 = time.time()
# for i in np.arange(num):
#     d[i] = np.exp(i)
# time1 = time.time()
# print("Time cost by np.arange without loop comprehension:", time1 - time0)

# x = []
# y = [0.46480801738564065, 0.4850665530003456, 0.47865774076080886, 0.38725703286483043, 0.1906325317335823, -0.081752420178889, -0.32229034048376376, -0.4516987136095837, -0.45383219076481396, -0.32795905398129266, -0.08386529306880434, 0.22149423707579874, 0.486542749208417, 0.6551835454428103, 0.745020148296143, 0.7832019549132729]
# for i in range(16):
#     x.append(-1 + 0.2 * i)
# plt.plot(x, y)
# plt.show()

size = 80
mat1 = matrix_numpy(size, size)
mat2 = matrix_numpy(size, size)

time0 = time.perf_counter()
temp = [None] * 2501
for i in range(2500, -1, -1):
    if i < 25:
        temp[i] = mat1
    else:
        temp[i] = mat2
time1 = time.perf_counter()
print(time1 - time0)

time0 = time.perf_counter()
temp = [mat1 if i < 25 else mat2 for i in np.arange(2500, -1, -1)]
time1 = time.perf_counter()
print(time1 - time0)

time0 = time.perf_counter()
temp = [mat1 if i < 25 else mat2 for i in range(2500, -1, -1)]
time1 = time.perf_counter()
print(time1 - time0)

time0 = time.perf_counter()
temp = [None] * 2501
for i in np.arange(2500, -1, -1):
    if i < 25:
        temp[i] = mat1
    else:
        temp[i] = mat2
time1 = time.perf_counter()
print(time1 - time0)

time0 = time.perf_counter()
temp = [None] * 2501
for i in range(2500, -1, -1):
    if i < 25:
        temp[i] = mat1
    else:
        temp[i] = mat2
time1 = time.perf_counter()
print(time1 - time0)
