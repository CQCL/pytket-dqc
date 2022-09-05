from __future__ import annotations

from typing import TYPE_CHECKING, List, Tuple

if TYPE_CHECKING:
    from pytket_dqc.networks import NISQNetwork
    from pytket_dqc.circuits import HypergraphCircuit

from networkx.algorithms.approximation.steinertree import (  # type: ignore
    steiner_tree
)
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
        """Overrides the default implementation"""
        if isinstance(other, Placement):
            return self.placement == other.placement
        return False

    def __str__(self):
        return str(self.placement)

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

    def cost(
        self,
        circuit: HypergraphCircuit,
        network: NISQNetwork
    ) -> int:
        """Cost of placement of ``circuit`` onto ``network``. The cost is
        measured as the number of e-bits which would be required.

        :param circuit: Circuit placed onto ``network`` by placement.
        :type circuit: HypergraphCircuit
        :param network: Network onto which ``circuit`` is placed by placement.
        :type network: NISQNetwork
        :raises Exception: Raised if this is not a valid placement of
            ``circuit`` onto ``network``.
        :return: Cost, in e-bits required, of this placement of ``circuit``
            onto ``network``.
        :rtype: int
        """

        cost = 0
        if self.is_valid(circuit, network):
            for hyperedge in circuit.hyperedge_list:
                # Cost of distributing gates in a hyperedge corresponds
                # to the number of edges in steiner tree connecting all
                # servers used by vertices in hyperedge.

                dist_graph = self.get_distribution_tree(
                    hyperedge.vertices,
                    circuit.get_qubit_vertex(hyperedge),
                    network
                )
                cost += len(dist_graph) * hyperedge.weight
        else:
            raise Exception("This is not a valid placement.")

        return cost

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
