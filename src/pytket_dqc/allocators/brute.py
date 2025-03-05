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

from pytket_dqc.allocators import Allocator
import itertools
from pytket_dqc.placement import Placement
from pytket_dqc.circuits import HypergraphCircuit, Distribution

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pytket import Circuit
    from pytket_dqc.networks import NISQNetwork


class Brute(Allocator):
    """Brute force allocator which searches through all placements
    for the best one.
    """

    def __init__(self) -> None:
        pass

    def allocate(self, circ: Circuit, network: NISQNetwork, **kwargs) -> Distribution:
        """Distribute quantum circuit by looking at all possible placements
        and returning the one with the lowest cost.

        :param circ: Circuit to distribute.
        :type circ: pytket.Circuit
        :param network: Network onto which ``circ`` should be distributed.
        :type network: NISQNetwork
        :raises Exception: Raised if no valid placement could be found.
        :return: Distribution of ``circ`` onto ``network``.
        :rtype: Distribution
        """

        dist_circ = HypergraphCircuit(circ)
        if not network.can_implement(dist_circ):
            raise Exception("This circuit cannot be implemented on this network.")

        # List of all vertices to be placed
        vertex_list = dist_circ.vertex_list
        # List of all servers which could be used
        server_list = network.get_server_list()

        # Initialise list of all valid placements
        valid_placement_list = []

        # TODO: It would be preferable to only check placement which are valid.
        # It may also be more memory efficient to not store a list of all
        # valid placements but to check their cost as they are generated.

        # Iterate over all placements, even those that are not valid.
        # Determin if they are valid, and add them to list if so.
        for placement_set in itertools.product(server_list, repeat=len(vertex_list)):
            placement_list = list(placement_set)

            # build placement from list of vertices and servers.
            placement_dict = {
                vertex: server for vertex, server in zip(vertex_list, placement_list)
            }
            placement = Placement(placement_dict)

            # Append to list is placement is valid.
            if placement.is_valid(dist_circ, network):
                valid_placement_list.append(placement)

        # Raise exception if there are no valid placements. This could happen
        # if the network is too small for example.
        if len(valid_placement_list) == 0:
            raise Exception("No valid placement could be found.")

        # Initialise minimum placement cost to be that of the first
        # valid placement.
        minimum_cost_distribution = Distribution(
            dist_circ, valid_placement_list[0], network
        )
        minimum_distribution_cost = minimum_cost_distribution.cost()
        # Check if any of the other valid placements have smaller cost.
        for placement in valid_placement_list[1:]:
            distribution = Distribution(dist_circ, placement, network)
            distribution_cost = distribution.cost()
            if distribution_cost < minimum_distribution_cost:
                minimum_cost_distribution = distribution
                minimum_distribution_cost = distribution_cost
            if distribution_cost == 0:
                break

        assert minimum_cost_distribution.is_valid()
        return minimum_cost_distribution
