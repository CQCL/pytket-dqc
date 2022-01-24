from __future__ import annotations

from pytket_dqc.distributors import Distributor
from pytket_dqc.placement import Placement
from typing import TYPE_CHECKING
import random

if TYPE_CHECKING:
    from pytket_dqc import DistributedCircuit
    from pytket_dqc.networks import NISQNetwork


def order_reducing_size(
    server_qubits: dict[int, list[int]]
) -> dict[int, list[int]]:

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


class Annealing(Distributor):
    def __init__(self):
        pass

    def distribute(
        self,
        dist_circ: DistributedCircuit,
        network: NISQNetwork,
        seed: int = None,
        interations: int = 10000
    ) -> Placement:

        # Get a naive initial placement of the vertices onto the servers.
        vertex_server_placement = self.initial_placement(dist_circ, network)

        placement_cost = vertex_server_placement.cost(
            dist_circ, network)
        print("placement_cost", placement_cost)

        return vertex_server_placement

    def random_initial_placement(
        self,
        dist_c: DistributedCircuit,
        network: NISQNetwork,
        seed: int = None
    ) -> Placement:

        # TODO: Add test that the number of qubits is big enough

        random.seed(seed)

        placement_dict: dict[int, int] = {}

        server_list = network.get_server_list()

        for vertex in dist_c.vertex_list:

            server_full = True

            while server_full:

                random_server = random.choice(server_list)

                vertices_places = [
                    placed_v
                    for placed_v, server in placement_dict.items()
                    if server == random_server
                ]
                vertices_places = [
                    placed_v
                    for placed_v in vertices_places
                    if dist_c.vertex_circuit_map[placed_v]['type'] == 'qubit'
                ]

                server_full = (len(vertices_places) == len(
                    network.server_qubits[random_server]))

            placement_dict[vertex] = random_server

        placement = Placement(placement_dict)
        assert placement.is_placement(dist_c, network)

        return placement

    def initial_placement(
        self,
        dist_circ: DistributedCircuit,
        network: NISQNetwork
    ) -> Placement:

        # A map from hypergraph vertices to the server on which
        # they are placed.
        vertex_server_map: dict[int, int] = {}

        # Order the servers so that the larger ones are used first
        server_qubits = order_reducing_size(network.server_qubits)

        # A list of all the vertices in the hypergraph which correspond to
        # qubits in the original circuit.
        qubit_vertex_list = [
            vertex
            for vertex, vertex_info in dist_circ.vertex_circuit_map.items()
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
        gate_vertex_list = [
            vertex
            for vertex, vertex_info in dist_circ.vertex_circuit_map.items()
            if vertex_info['type'] == 'gate'
        ]

        # Assign all gate vertices to the first server.
        first_server = list(server_qubits.keys())[0]
        for vertex in gate_vertex_list:
            vertex_server_map[vertex] = first_server

        placement = Placement(vertex_server_map)

        assert placement.is_placement(dist_circ, network)

        return placement
