# Copyright 2023 Quantinuum
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

from .refiner import Refiner
from pytket_dqc.circuits.distribution import Distribution


class SequenceRefiner(Refiner):
    """Performs sequence of :class:`.Refiner` as provided by the user.
    """

    def __init__(self, refiner_list: list[Refiner]):
        """SequenceRefiner is initialised with a list of :class:`.Refiner`
        to be run in sequence.

        :param refiner_list: List of Refiners to be run in sequence.
        :type refiner_list: list[Refiner]
        """

        self.refiner_list = refiner_list

    def refine(self, distribution: Distribution, **kwargs) -> bool:
        """Perform each of the refinements in the provided sequence.

        :param distribution: Distribution to be refined.
        :type distribution: Distribution
        :return: True if if any of the Refiners
            in the sequence makes a refinement. False otherwise.
        :rtype: bool
        """

        refinement_made = False
        for refiner in self.refiner_list:
            refinement_made |= refiner.refine(distribution)

        return refinement_made
