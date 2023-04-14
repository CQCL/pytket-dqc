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

from .refiner import Refiner
from pytket_dqc.circuits.distribution import Distribution


class RepeatRefiner(Refiner):
    """Repeats given :class:`.Refiner` until no refinement is made.
    """

    def __init__(self, refiner: Refiner):
        """RepeatRefiner is initialised with :class:`.Refiner` to repeat.

        :param refiner: Refiner to repeat.
        :type refiner: Refiner
        """

        self.refiner = refiner

    def refine(self, distribution: Distribution, **kwargs) -> bool:
        """Repeat given :class:`.Refiner` until it makes no more refinements.

        :param distribution: Distribution to be refined.
        :type distribution: Distribution
        :return: True if at least one action of the repeated Refiner
            makes a refinement. False otherwise.
        :rtype: bool
        """

        refinement_made = self.refiner.refine(distribution)
        one_refinement_made = refinement_made
        while refinement_made:
            refinement_made = self.refiner.refine(distribution)

        return one_refinement_made
