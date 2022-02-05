from __future__ import annotations

from pytket_dqc.distributors import Distributor
from pytket_dqc.placement import Placement
from typing import TYPE_CHECKING, Union
import random
from .random import Random
from .ordered import Ordered

if TYPE_CHECKING:
    from pytket_dqc import DistributedCircuit
    from pytket_dqc.networks import NISQNetwork


class Annealing(Distributor):
    def __init__(self):
        pass

    def distribute(
        self,
        dist_circ: DistributedCircuit,
        network: NISQNetwork,
        **kwargs
    ) -> Placement:

        seed = kwargs.get("seed", None)
        iterations = kwargs.get("iterations", 10000)
        place_method = kwargs.get("initial_place_method", 'ordered')

        random.seed(seed)

        initial_distributor: Union[Ordered, Random]
        if place_method == 'ordered':
            initial_distributor = Ordered()
        elif place_method == 'random':
            initial_distributor = Random()
        else:
            raise Exception(
                f"'{place_method}' is not a valid initial placement method")
        placement = initial_distributor.distribute(dist_circ, network)
        cost = placement.cost(dist_circ, network)

        # TODO: Check that the initial placement does not have cost 0, and
        # that not all qubits are already in the same server etc.

        for _ in range(iterations):

            if cost == 0:
                break

            vertex_to_move = random.choice(dist_circ.vertex_list)

            home_server = placement.placement[vertex_to_move]

            # Pick a random server to move to
            possible_servers = network.get_server_list()
            possible_servers.remove(placement.placement[vertex_to_move])
            destination_server = random.choice(possible_servers)

            swap_placement_dict = placement.placement.copy()

            if dist_circ.vertex_circuit_map[vertex_to_move]['type'] == 'qubit':

                # List qubit vertices
                destination_server_qubit_list = [
                    v for v in dist_circ.vertex_list
                    if (dist_circ.vertex_circuit_map[v]['type'] == 'qubit')
                ]

                # remove qubits in the same server
                destination_server_qubit_list = [
                    v for v in destination_server_qubit_list
                    if (placement.placement[v] == destination_server)
                ]

                q_in_dest = len(destination_server_qubit_list)
                size_dest = len(network.server_qubits[destination_server])

                if q_in_dest == size_dest:
                    swap_vertex = random.choice(destination_server_qubit_list)
                    swap_placement_dict[swap_vertex] = home_server

            swap_placement_dict[vertex_to_move] = destination_server

            swap_placement = Placement(swap_placement_dict)
            swap_cost = swap_placement.cost(dist_circ, network)

            if swap_cost < cost:
                placement = swap_placement
                cost = swap_cost

            assert placement.is_placement(dist_circ, network)

        return placement
