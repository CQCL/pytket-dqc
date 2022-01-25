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
        iterations: int = 10000
    ) -> Placement:

        random.seed(seed)

        # Get a naive initial placement of the vertices onto the servers.
        placement = self.initial_placement(dist_circ, network)
        cost = placement.cost(dist_circ, network)
        print("initial placement", placement.placement)
        print("placement_cost", cost)

        # TODO: Check that the initial placement does not have cost 0, and
        # that not all qubits are already in the same server etc.

        print("===============")

        for i in range(iterations):

            print("iteration:", i)

            vertex = random.choice(dist_circ.vertex_list)
            print("vertex chosen:", vertex)

            print("vertex type:", dist_circ.vertex_circuit_map[vertex]['type'])

            server = placement.placement[vertex]

            print("vertex server:", server)

            # TODO: This is set up to swap qubit vertices. Change it so that 
            # empty spaces in servers which have at least one qubit in them 
            # can also be used.

            if dist_circ.vertex_circuit_map[vertex]['type'] == 'qubit':

                # List qubit vertices
                qubit_list = [
                    qubit_vertex 
                    for qubit_vertex in dist_circ.vertex_list 
                    if (dist_circ.vertex_circuit_map[qubit_vertex]['type'] == 'qubit') 
                ]
                print("all qubit vertices:", qubit_list)
                # Remove chosen vertex
                qubit_list.remove(vertex)
                # remove qubits in the same server
                qubit_list = [
                    qubit_vertex 
                    for qubit_vertex in qubit_list 
                    if (not (placement.placement[qubit_vertex] == server))
                ]
                print("qubit_list:", qubit_list)

                assert len(qubit_list) >= 1

                swap_vertex = random.choice(qubit_list)
                swap_server = placement.placement[swap_vertex]

                print("swap_vertex", swap_vertex)

                swap_placement_dict = placement.placement.copy()
                swap_placement_dict[vertex] = swap_server
                swap_placement_dict[swap_vertex] = server

                print("swap_placement_dict", swap_placement_dict)

                swap_placement = Placement(swap_placement_dict)
                swap_cost = swap_placement.cost(dist_circ, network)

                print("swap_cost", swap_cost)

                if swap_cost < cost:
                    placement = swap_placement
                    cost = swap_cost


                



            elif dist_circ.vertex_circuit_map[vertex]['type'] == 'gate':
                pass

            else:
                raise Exception("Type not recognised")

            print("===== iteration end =====")
        return placement

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
