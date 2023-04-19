# Copyright 2023 Quantinuum and The University of Tokyo
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

from abc import ABC, abstractmethod
from pytket_dqc.circuits.distribution import Distribution


class Refiner(ABC):
    """Abstract base class defining the behaviors of Refiners, which perform
    in place processing on a :class:`.Distribution`.
    """

    @abstractmethod
    def refine(self, distribution: Distribution, **kwargs) -> bool:
        """Perform in place refinement of a :class:`.Distribution`.

        :param distribution: Distribution to be refined.
        :type distribution: Distribution
        :return: True if a refinement has been performed. False otherwise.
        :rtype: bool
        """
        pass
