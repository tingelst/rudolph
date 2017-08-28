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

from .quaternion import Quaternion
from .dual_quaternion import DualQuaternion, recover

Q = np.diag([0, 0, 0, 0, 0, 0, 0.0011, 0.0011,
             0.0011, 0.00002, 0.00002, 0.00002])

R = np.diag([0.0000003513, 0.00000259, 0.000003196,
             0.00000547, 0.00000498, 0.0001018])

G = np.eye(12)
G[:6, :6] *= -0.5

H = np.zeros((6, 12))
H[:6, :6] = np.eye(6).astype(np.float)


def get_F(w):
    F = np.zeros((12, 12))
    F[:6, :6] = -w
    F[:6, 6:] = np.eye(6) * -0.5
    return F


class State(object):
    def __init__(self, x=None, dx=None):
        self.x = x
        self.dx = dx


def time_propagation(P, X, t):

    dq = X.x.conj * X.dx * -0.5
    q = (X.x + dq * t).normalized()

    F = get_F(X.dx.vskew)

    dP = F @ P + P @ F.T + G @ Q @ G.T

    P += t * dP

    return State(q, X.dx), P


def measurement_update(meas, P, X):
    K = P @ H.T @ np.linalg.inv(H @ P @ H.T + R)
    err = X.x.conj * meas
    err_red = err.v
    delta_err = K @ err_red
    b = delta_err[6:]
    X.x = X.x * recover(delta_err[:6])
    X.dx = X.dx + DualQuaternion(Quaternion(0, b[0], b[1], b[2]),
                                 Quaternion(0, b[3], b[4], b[5]))
    
    dum = np.eye(12) - K @ H
    P = dum @ P @ dum.T + K @ R @ K.T

    return X, P


