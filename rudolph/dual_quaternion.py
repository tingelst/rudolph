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

class DualQuaternion(object):
    def __init__(self, qreal, qdual):
        self._real = qreal
        self._dual = qdual
        
    @property
    def real(self):
        return self._real
    
    @property
    def dual(self):
        return self._dual
    
    def __repr__(self):
        return self._real.__repr__() + '\n' + self._dual.__repr__()
    
    def normalized(self):
        return DualQuaternion(self.real.normalized(), 
                              Quaternion(*np.dot(np.eye(4) - np.outer(self.real._data, self.real._data) / self.real.norm()**2, 
                                                 self.dual._data.reshape(4,1))))
    @property
    def conj(self):
        return DualQuaternion(self.real.conj, self.dual.conj)
    
    def __mul__(self, other):
        return DualQuaternion(self.real * other.real, self.real * other.dual + self.dual * other.real)