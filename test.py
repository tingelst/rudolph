import numpy as np

from rudolph import Quaternion as Quat
from rudolph import DualQuaternion as DQuat


d1 = DQuat(Quat(1.,2.,3.,4.), Quat(5.,6.,7.,8.))
d2 = DQuat(Quat(1.,2.,3.,4.), Quat(5.,6.,7.,8.))

d3 = (d1 * d2).normalized()

print(d3 * d3.conj)
