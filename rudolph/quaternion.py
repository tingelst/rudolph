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

import numpy as np
np.set_printoptions(precision=4)


def skew(v):
    return np.array([[0.0, -v[2], v[1]],
                     [v[2], 0.0, -v[0]],
                     [-v[1], v[0], 0.0]])


class Quaternion(object):
    def __init__(self, q0=0, q1=0, q2=0, q3=0):
        self._data = np.array([q0, q1, q2, q3]).flatten()

    @property
    def left_matrix(self):
        q0, q1, q2, q3 = self._data
        matrix = np.array([[q0, -q1, -q2, -q3],
                           [q1,  q0, -q3,  q2],
                           [q2,  q3,  q0, -q1],
                           [q3, -q2,  q1,  q0]])
        return matrix

    @property
    def right_matrix(self):
        q0, q1, q2, q3 = self._data
        matrix = np.array([[q0, -q1, -q2, -q3],
                           [q1,  q0,  q3, -q2],
                           [q2, -q3,  q0,  q1],
                           [q3,  q2, -q1,  q0]])
        return matrix

    @property
    def s(self):
        return self._data[0]

    @property
    def v(self):
        return self._data[1:]

    @property
    def vskew(self):
        return skew(self.v)

    def __neg__(self):
        return Quaternion(*(self._data * -1.0))

    def __mul__(self, other):
        if isinstance(other, (float, np.float)):
            other = Quaternion(other, 0, 0, 0)
        return Quaternion(*np.dot(self.left_matrix, other._data.reshape(4, 1)))

    def __add__(self, other):
        return Quaternion(*(self._data + other._data))

    def __add__(self, other):
        return Quaternion(*(self._data + other._data))

    @property
    def conj(self):
        return Quaternion(self._data[0], -self._data[1], -self._data[2], -self._data[3])

    def spin(self, other):
        return self * other * self.conj

    def norm(self):
        return np.linalg.norm(self._data)

    def normalized(self):
        self._data /= self.norm()
        return self

    def __repr__(self):
        return '[{}, {}, {}, {}]'.format(*self._data)

    def asarray(self):
        return self._data
