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

from __future__ import annotations

from abc import ABC, abstractmethod

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pytket_dqc.circuits.distribution import Distribution
    from pytket_dqc.networks import NISQNetwork
    from pytket import Circuit


class Distributor(ABC):
    """Abstract class defining the structure of distributors. Distributors
    are complete recommended DQC workflows.
    """

    def __init__(self) -> None:
        pass

    @abstractmethod
    def distribute(
        self, circ: Circuit, network: NISQNetwork, **kwargs
    ) -> Distribution:
        """Abstract method producing a distribution of the given circuit
        onto the given network.

        :param circ: Circuit to be distributed
        :type circ: Circuit
        :param network: Network onto which circuit should be distributed
        :type network: NISQNetwork
        :return: Distribution of circ onto network.
        :rtype: Distribution
        """
        pass
