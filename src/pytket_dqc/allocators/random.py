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
from pytket_dqc.placement import Placement
from pytket_dqc.circuits import HypergraphCircuit, Distribution
from typing import TYPE_CHECKING
import random

if TYPE_CHECKING:
    from pytket import Circuit
    from pytket_dqc.networks import NISQNetwork


class Random(Allocator):
    """Distribute hypergraph vertices onto servers at random. The resulting
    placement is valid, which is to say vertices will not be placed on servers
    once that are full.
    """

    def __init__(self) -> None:
        pass

    def allocate(self, circ: Circuit, network: NISQNetwork, **kwargs) -> Distribution:
        """Distribute ``circ`` onto ``network`` by randomly placing
        vertices onto servers. Qubit vertices are placed onto servers
        until the server is full. Gate vertices are placed on servers at random
        without restriction.

        :param circ: Circuit to distribute.
        :type circ: pytket.Circuit
        :param network: Network onto which ``circ`` should be distributed.
        :type network: NISQNetwork
        :return: Distribution of ``circ`` onto ``network``.
        :rtype: Distribution
        """

        dist_circ = HypergraphCircuit(circ)
        if not network.can_implement(dist_circ):
            raise Exception("This circuit cannot be implemented on this network.")

        seed = kwargs.get("seed", None)
        random.seed(seed)

        # TODO: Add test that the number of qubits is big enough. If the number
        # of qubits is too big then the while loop below will not complete.

        # Initialise dictionary from hypergraph vertices to servers
        placement_dict: dict[int, int] = {}

        server_list = network.get_server_list()

        # Place each vertex in sequence.
        for vertex in dist_circ.vertex_list:
            # TODO: Instead of this, maintain a list of servers which are not
            # full.
            server_full = True

            # Pick random servers util an empty one is found.
            while server_full:
                # Choose random server onto which to place vertex.
                random_server = random.choice(server_list)

                # Get list of vertices placed onto server.
                vertices_places = [
                    placed_v
                    for placed_v, server in placement_dict.items()
                    if server == random_server
                ]
                # Get list of qubit vertices placed onto server.
                vertices_places = [
                    v
                    for v in vertices_places
                    if dist_circ._vertex_circuit_map[v]["type"] == "qubit"
                ]

                # Check if server is full
                server_full = len(vertices_places) == len(
                    network.server_qubits[random_server]
                )

            placement_dict[vertex] = random_server

        placement = Placement(placement_dict)
        assert placement.is_valid(dist_circ, network)

        return Distribution(dist_circ, placement, network)
