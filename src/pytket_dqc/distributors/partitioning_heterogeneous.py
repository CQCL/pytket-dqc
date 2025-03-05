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

from .distributor import Distributor
from pytket_dqc import NISQNetwork, Distribution
from pytket_dqc.refiners import (
    RepeatRefiner,
    EagerHTypeMerge,
    BoundaryReallocation,
)
from pytket_dqc.allocators import HypergraphPartitioning, Annealing
from pytket import Circuit


class PartitioningAnnealing(Distributor):
    """Distributor using the :class:`.Annealing` allocator."""

    def distribute(self, circ: Circuit, network: NISQNetwork, **kwargs) -> Distribution:
        """Method producing a distribution of the given circuit
        onto the given network.

        Note that kwargs are passed on to the allocate method of
        :class:`.Annealing`.

        :param circ: Circuit to be distributed
        :type circ: Circuit
        :param network: Network onto which circuit should be distributed
        :type network: NISQNetwork
        :return: Distribution of circ onto network.
        :rtype: Distribution
        """

        return Annealing().allocate(circ, network, **kwargs)


class PartitioningHeterogeneous(Distributor):
    """Distributor refining the output of :class:`.HypergraphPartitioning`
    to adapt the result to heterogeneous networks.
    """

    def distribute(self, circ: Circuit, network: NISQNetwork, **kwargs) -> Distribution:
        """Method producing a distribution of the given circuit
        onto the given network.

        Note that kwargs are passed on to the allocate method of
        :class:`.HypergraphPartitioning` and the `refine` method of
        :class:`.BoundaryReallocation`.

        :param circ: Circuit to be distributed
        :type circ: Circuit
        :param network: Network onto which circuit should be distributed
        :type network: NISQNetwork
        :return: Distribution of circ onto network.
        :rtype: Distribution
        """

        distribution = HypergraphPartitioning().allocate(circ, network, **kwargs)
        refiner = BoundaryReallocation()
        refiner.refine(distribution, **kwargs)

        return distribution


class PartitioningHeterogeneousEmbedding(Distributor):
    """Distributor refining the output of :class:`.PartitioningHeterogeneous`
    to make use of embedding.
    """

    def distribute(self, circ: Circuit, network: NISQNetwork, **kwargs) -> Distribution:
        """Method producing a distribution of the given circuit
        onto the given network.

        Note that kwargs are passed on to the distribute method of
        :class:`.PartitioningHeterogeneousEmbedding`.

        :param circ: Circuit to be distributed
        :type circ: Circuit
        :param network: Network onto which circuit should be distributed
        :type network: NISQNetwork
        :return: Distribution of circ onto network.
        :rtype: Distribution
        """

        initial_distributor = kwargs.get(
            "initial_distributor", PartitioningHeterogeneous()
        )

        distribution = initial_distributor.distribute(circ, network, **kwargs)
        refiner = RepeatRefiner(EagerHTypeMerge())
        refiner.refine(distribution)

        return distribution
