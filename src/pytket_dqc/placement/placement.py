from __future__ import annotations

from typing import TYPE_CHECKING, cast
if TYPE_CHECKING:
    from pytket_dqc.networks import NISQNetwork
    from pytket_dqc.circuits import DistributedCircuit

import networkx as nx  # type: ignore


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

            G = network.get_server_nx()

            for hyperedge in circuit.hyperedge_list:
                # Generate a list of where each vertex of the hyperedge
                # is placed
                vertex_list = cast(list[int], hyperedge['hyperedge'])
                hyperedge_placement = [
                    self.placement[vertex] for vertex in vertex_list
                ]

                # Find the server where the qubit vertex is placed
                qubit_vertex_list = [
                    vertex
                    for vertex in vertex_list
                    if circuit.vertex_circuit_map[vertex]['type'] == 'qubit'
                ]
                assert len(qubit_vertex_list) == 1
                qubit_vertex = qubit_vertex_list[0]
                qubit_vertex_server = self.placement[qubit_vertex]

                # The cost is equal to the distance between each of the
                # vertices and the qubit vertex.
                # TODO: This approach very naively assumes that the control is
                # teleported back when a new server pair is interacted.
                # There may be a better approach.
                unique_servers_used = list(set(hyperedge_placement))
                for server in unique_servers_used:
                    shortest_path_length = nx.shortest_path_length(
                        G, qubit_vertex_server, server
                    )
                    cost += shortest_path_length * hyperedge['weight']

        else:
            raise Exception("This is not a valid placement.")

        return cost
