import numpy as np

import matplotlib.pyplot as plt

from rudolph.quaternion import skew
from rudolph.dual_quaternion import recover
from rudolph import Quaternion as Quat
from rudolph import DualQuaternion as DQuat

from rudolph.dqmekf import (G, time_propagation, State, measurement_update)

timestamps = np.loadtxt('../datasets/timestamps.txt')
estimates = np.loadtxt(
    '../datasets/icp_estimates_withinit_NN10_icp10.txt').reshape(-1, 8)

X = State()
X.x = DQuat(Quat(1, 0, 0, 0), Quat(0, 0, 0, 0))
X.dx = DQuat()

P = 10e-9 * np.eye(12)

t = np.loadtxt('../datasets/timestamps.txt')
q_icp = np.loadtxt(
    '../datasets/icp_estimates_withinit_NN10_icp10.txt').reshape(-1, 8)

a = []

filts2 = np.loadtxt('../../DQ_filter_lib/filts.txt')
xyzs = []
for f in filts2:
     xyzs.append(DQuat(Quat(*f[:4]), Quat(*f[4:8])).trs().v.flatten())

xyzs = np.array(xyzs).reshape(-1,3)

measts = []
filts = []

t_s = t[0]
for x in range(1, 197):
    time_propagation(P, X, t[x] - t_s)
    q = q_icp[x - 1]
    meas = DQuat(Quat(*q[:4]), Quat(*q[4:]))
    measurement_update(meas, P, X)


    filts.append(X.x.trs().v)
    measts.append(meas.trs().v)

    a.append(X.x.asarray().flatten())

    t_s = t[x]

a = np.array(a).reshape(-1, 8)

f, axarr = plt.subplots(4, sharex=True)
for i in range(4):
    axarr[i].plot(q_icp[:, i])
    axarr[i].plot(a[:, i])

measts = np.array(measts).reshape(-1,3) * 1000
filts = np.array(filts).reshape(-1,3) * 1000

f1, axarr1 = plt.subplots(3, sharex=True)
for i in range(3):
    axarr1[i].plot(measts[:, i])
    axarr1[i].plot(filts[:, i])
    # axarr1[i].plot(xyzs[:,i])

plt.show()
