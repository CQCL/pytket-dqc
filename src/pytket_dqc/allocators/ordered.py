from __future__ import annotations

from pytket_dqc.allocators import Allocator
from pytket_dqc.placement import Placement
from pytket_dqc.circuits import HypergraphCircuit, Distribution
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pytket import Circuit
    from pytket_dqc.networks import NISQNetwork


def order_reducing_size(
    server_qubits: dict[int, list[int]]
) -> dict[int, list[int]]:
    """Reorder ``server_qubits`` dictionary so that servers are in
    reducing size.

    :param server_qubits: Dictionary mapping servers to the qubits
        they contain.
    :type server_qubits: dict[int, list[int]]
    :return: Dictionary mapping servers to the qubits they contain,
        in order from largest to smallest server.
    :rtype: dict[int, list[int]]
    """

    # List of servers whose position in a ranking by size is unknown
    position_unknown = list(server_qubits.keys())

    # The order of the servers by their size
    order = []

    # Look through the list of servers n-1 times, where n is the number of
    # servers, for the largest. Add the largest found to the order list.
    for _ in range(len(server_qubits)-1):
        # Initialise with largest at start of list of servers with unknown
        # position in the order by size.
        largest_i = 0
        largest = position_unknown[largest_i]

        # For each server with unknown position, check if it is the largest.
        for i, server in enumerate(position_unknown[1:]):
            if len(server_qubits[largest]) < len(server_qubits[server]):
                largest = server
                largest_i = i + 1

        # Append the largest found to the order list, and remove it from the
        # list of servers with unknown position.
        order.append(largest)
        position_unknown.pop(largest_i)

    # Add the remaining server to the order. Only one will remain.
    order.append(position_unknown[0])

    return {server: server_qubits[server] for server in order}


class Ordered(Allocator):
    """Distribute hypergraph vertices onto servers, populating the largest
    servers first until all vertices are assigned.
    """
    def __init__(self) -> None:
        pass

    def allocate(
        self,
        circ: Circuit,
        network: NISQNetwork,
        **kwargs
    ) -> Distribution:
        """Distribute ``circ`` onto ``network`` by placing quibts onto
        servers, in decreasing order of size, until they are full.

        :param circ: Circuit to distribute.
        :type circ: pytket.Circuit
        :param network: Network onto which ``circ`` should be distributed.
        :type network: NISQNetwork
        :return: Distribution of ``circ`` onto ``network``.
        :rtype: Distribution
        """

        dist_circ = HypergraphCircuit(circ)
        if not network.can_implement(dist_circ):
            raise Exception(
                "This circuit cannot be implemented on this network."
                )

        # Initialise a map from hypergraph vertices to the server on which
        # they are placed.
        vertex_server_map: dict[int, int] = {}

        # Order the servers so that the larger ones are used first
        server_qubits = order_reducing_size(network.server_qubits)

        # A list of all the vertices in the hypergraph which correspond to
        # qubits in the original circuit.
        # TODO: Turn this into a method of HypergraphCircuit as
        # it is used often.
        qubit_vertex_list = [
            vertex
            for vertex, vertex_info in dist_circ._vertex_circuit_map.items()
            if vertex_info['type'] == 'qubit'
        ]

        for server, qubit_list in server_qubits.items():
            # Assign the first n qubit vertices to server, where
            # server has n vertices. This ensures that each server does not
            # have more qubits than it cas handle.
            for vertex in qubit_vertex_list[:len(qubit_list)]:
                vertex_server_map[vertex] = server
            # Remove the vertices which have been assigned to a server.
            qubit_vertex_list = qubit_vertex_list[len(qubit_list):]

        # A list of all the vertices in the hypergraph which correspond to
        # gate in the original circuit.
        # TODO: Turn this into a method of HypergraphCircuit as
        # it is used often.
        gate_vertex_list = [
            vertex
            for vertex, vertex_info in dist_circ._vertex_circuit_map.items()
            if vertex_info['type'] == 'gate'
        ]

        # Assign all gate vertices to the first server.
        # TODO: This can certainly be improved. Possible assign gates to
        # the same server as one of it's target qubits.
        first_server = list(server_qubits.keys())[0]
        for vertex in gate_vertex_list:
            vertex_server_map[vertex] = first_server

        placement = Placement(vertex_server_map)

        assert placement.is_valid(dist_circ, network)

        return Distribution(dist_circ, placement, network)
