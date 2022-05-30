from __future__ import annotations

from typing import TYPE_CHECKING, cast, List
if TYPE_CHECKING:
    from pytket_dqc.networks import NISQNetwork
    from pytket_dqc.circuits import DistributedCircuit

import networkx as nx  # type: ignore
from networkx.algorithms.approximation.steinertree import (  # type: ignore
    steiner_tree
)


class Placement:
    """Placement of hypergraph onto server network.

    :param placement: Dictionary mapping hypergraph vertices to
        server indexes.
    :type placement: dict[int, int]
    """

    def __init__(self, placement: dict[int, int]):
        self.placement = placement

    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(other, Placement):
            return self.placement == other.placement
        return False

    def is_placement(
        self,
        circuit: DistributedCircuit,
        network: NISQNetwork
    ) -> bool:
        """Check if placement is valid. In particular check that no more
        qubits are allotted to a server than can be accommodated.

        :param circuit: Circuit being placed onto ``network`` by placement.
        :type circuit: DistributedCircuit
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
        for server in list(set(self.placement.values())):
            vertices = [vertex for vertex in self.placement.keys()
                        if self.placement[vertex] == server]
            qubits = [
                vertex
                for vertex in vertices
                if circuit.vertex_circuit_map[vertex]['type'] == 'qubit'
            ]
            if len(qubits) > len(network.server_qubits[server]):
                is_valid = False

        return is_valid

    def cost(
        self,
        circuit: DistributedCircuit,
        network: NISQNetwork
    ) -> int:
        """Cost of placement of ``circuit`` onto ``network``. The cost is
        measured as the number of e-bits which would be required.

        :param circuit: Circuit placed onto ``network`` by placement.
        :type circuit: DistributedCircuit
        :param network: Network onto which ``circuit`` is placed by placement.
        :type network: NISQNetwork
        :raises Exception: Raised if this is not a valid placement of
            ``circuit`` onto ``network``.
        :return: Cost, in e-bits required, of this placement of ``circuit``
            onto ``network``.
        :rtype: int
        """

        cost = 0
        if self.is_placement(circuit, network):
            for hyperedge in circuit.hyperedge_list:
                # Cost of distributing gates in a hyperedge corresponds
                # to the number of edges in steiner tree connecting all
                # servers used by vertices in hyperedge.
                dist_graph = self._get_distribution_graph(
                    cast(List[int], hyperedge['hyperedge']), network)
                cost += len(dist_graph.edges()) * \
                    cast(int, hyperedge['weight'])
        else:
            raise Exception("This is not a valid placement.")

        return cost

    def _get_distribution_graph(
        self,
        hyperedge: list[int],
        network: NISQNetwork
    ) -> nx.Graph:
        """Returns graph representing the edges along which distribution
        operations should act. This is the steiner tree covering the servers
        used by the vertices in the hyper edge.

        :param hyperedge: Hyperedge for which distribution graph
        should be found.
        :type hyperedge: list[int]
        :param network: Network onto which hyper edge should be distributed.
        :type network: NISQNetwork
        :return: Graph of distribution operations.
        :rtype: nx.Graph
        """

        servers_used = [value for key,
                        value in self.placement.items() if key in hyperedge]
        server_graph = network.get_server_nx()
        return steiner_tree(server_graph, servers_used)
