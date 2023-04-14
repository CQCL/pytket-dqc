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

from typing import TYPE_CHECKING, List, Tuple

if TYPE_CHECKING:
    from pytket_dqc.networks import NISQNetwork
    from pytket_dqc.circuits import HypergraphCircuit

from pytket_dqc.utils import steiner_tree
from pytket_dqc.utils import direct_from_origin


class Placement:
    """Placement of hypergraph onto server network.

    :param placement: Dictionary mapping hypergraph vertices to
        server indexes.
    :type placement: dict[int, int]
    """

    def __init__(self, placement: dict[int, int]):
        self.placement = placement

    def __eq__(self, other):
        """Check equality based on equality of components"""
        if isinstance(other, Placement):
            return self.placement == other.placement
        return False

    def __str__(self):
        return str(self.placement)

    def to_dict(self) -> dict[int, int]:
        """Generate JSON serialisable dictionary for the Placement.

        :return: Dictionary serialisation of the Placement.
        :rtype: dict[int, int]
        """
        return self.placement

    @classmethod
    def from_dict(cls, placement_dict: dict[int, int]):
        """Generate ``Placement`` instance from JSON serialisable dictionary.

        :param placement_dict: JSON serialisable dictionary
            representation of the Placement.
        :type placement_dict: dict[int, int]
        :return: Placement instance constructed from placement_dict.
        :rtype: Placement
        """
        placement = {
            int(node): int(server) for node, server in placement_dict.items()
        }
        return cls(placement=placement)

    def is_valid(
        self,
        circuit: HypergraphCircuit,
        network: NISQNetwork
    ) -> bool:
        """Check if placement is valid. In particular check that no more
        qubits are allotted to a server than can be accommodated.

        :param circuit: Circuit being placed onto ``network`` by placement.
        :type circuit: HypergraphCircuit
        :param network: Network ``circuit`` is placed onto by placement.
        :type network: NISQNetwork
        :return: Is a valid placement.
        :rtype: bool
        """

        if not circuit.is_placement(self):
            return False
        elif not network.is_placement(self):
            return False
        else:
            is_valid = True

        # Check that no more qubits are allotted to a server than can be
        # accommodated.
        for server in network.server_qubits.keys():
            vertices = [vertex for vertex in self.placement.keys()
                        if self.placement[vertex] == server]
            qubits = [
                vertex
                for vertex in vertices
                if circuit.is_qubit_vertex(vertex)
            ]
            if len(qubits) > len(network.server_qubits[server]):
                is_valid = False

        return is_valid

    def get_distribution_tree(
        self,
        hyperedge: list[int],
        qubit_node: int,
        network: NISQNetwork,
    ) -> List[Tuple[int, int]]:
        """Returns tree representing the edges along which distribution
        operations should act. This is the steiner tree covering the servers
        used by the vertices in the hyper edge.

        :param hyperedge: Hyperedge for which distribution graph
            should be found.
        :type hyperedge: list[int]
        :param qubit_node: Node in hyperedge which corresponds to a qubit.
        :type qubit_node: int
        :param network: Network onto which hyper edge should be distributed.
        :type network: NISQNetwork
        :return: List of edges along which distribution gates should act,
            with the direction and order in this they should act.
        :rtype: List[List[int]]
        """

        servers_used = [value for key,
                        value in self.placement.items() if key in hyperedge]
        server_graph = network.get_server_nx()

        # The Steiner tree problem is NP-complete. Indeed the networkx
        # steiner_tree is solving a problem which gives an upper bound on
        # the size of the Steiner tree. Importantly it produces a deterministic
        # output, which we rely on. In particular we assume the call to this
        # function made when calculating costs gives the same output as the
        # call that is made when the circuit is built and outputted.
        steiner_server_graph = steiner_tree(server_graph, servers_used)
        qubit_server = self.placement[qubit_node]
        return direct_from_origin(steiner_server_graph, qubit_server)

    def get_vertices_in(self, server: int) -> list[int]:
        """Return the list of vertices placed in ``server``.
        """
        return [v for v, s in self.placement.items() if s == server]
