# Copyright 2017 Norwegian University of Science and Technology
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# import numpy as np

import numpy as np

from .quaternion import Quaternion
from .quaternion import skew


def recover(v):
    v = v.flatten()
    s = np.sqrt(1 - np.linalg.norm(v[:3])**2)
    ds = - np.inner(v[:3], v[3:]) / s
    dq = DualQuaternion(Quaternion(
        s, v[0], v[1], v[2]), Quaternion(ds, v[3], v[4], v[5]))
    return dq


class DualQuaternion(object):
    def __init__(self, qreal=Quaternion(), qdual=Quaternion()):
        self._real = qreal
        self._dual = qdual

    @property
    def real(self):
        return self._real

    @property
    def dual(self):
        return self._dual

    @property
    def left_matrix(self):
        m8x8 = np.zeros((8, 8))
        m8x8[:4, :4] = self.real.left_matrix
        m8x8[4:, :4] = self.dual.left_matrix
        m8x8[4:, 4:] = self.real.left_matrix
        return m8x8

    @property
    def right_matrix(self):
        m8x8 = np.zeros((8, 8))
        m8x8[:4, :4] = self.real.right_matrix
        m8x8[4:, :4] = self.dual.right_matrix
        m8x8[4:, 4:] = self.real.right_matrix
        return m8x8

    @property
    def s(self):
        return np.array([self.real.s, self.dual.s]).reshape(2, 1)

    @property
    def v(self):
        return np.array([self.real.v, self.dual.v]).reshape(6, 1)

    def inverse(self):
        r = self.real
        d = self.dual
        nsq = (r.conj * r).s
        return DualQuaternion(r.conj * (1.0 / nsq),
                              r.conj * d * r.conj * (-1.0 / nsq**2))

    def __invert__(self):
        return self.inverse()

    @property
    def vskew(self):
        m6x6 = np.zeros((6, 6))
        m6x6[:3, :3] = self.real.vskew
        m6x6[3:, :3] = self.dual.vskew
        m6x6[3:, 3:] = self.real.vskew
        return m6x6

    def __repr__(self):
        return 'DualQuaternion:\n\t''real: ' + self._real.__repr__() +\
            '\n\t' + 'dual: ' + self._dual.__repr__()

    def normalized(self):
        dq = DualQuaternion
        return DualQuaternion(self.real.normalized(),
                              Quaternion(*np.dot(np.eye(4) - np.outer(self.real._data, self.real._data) / self.real.norm()**2,
                                                 self.dual._data.reshape(4, 1))))

    @property
    def conj(self):
        return DualQuaternion(self.real.conj, self.dual.conj)

    def __mul__(self, other):
        if isinstance(other, (float, np.float64)):
            other = DualQuaternion(Quaternion(other, 0, 0, 0), Quaternion())
        return DualQuaternion(self.real * other.real,
                              self.real * other.dual + self.dual * other.real)

    def __add__(self, other):
        return DualQuaternion(self.real + other.real, self.dual + other.dual)

    def __neg__(self):
        return DualQuaternion(-self.real, -self.dual)

    def trs(self):
        return self.dual * self.real.conj * 2.0

    def asarray(self):
        return np.concatenate((self.real.asarray(), self.dual.asarray())).reshape(8, 1)
