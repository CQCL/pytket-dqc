from __future__ import annotations

from pytket_dqc.distributors import Distributor
from pytket_dqc.placement import Placement
from typing import TYPE_CHECKING
import random

if TYPE_CHECKING:
    from pytket_dqc import DistributedCircuit
    from pytket_dqc.networks import NISQNetwork


class Random(Distributor):
    def __init__(self):
        pass

    def distribute(
        self,
        dist_circ: DistributedCircuit,
        network: NISQNetwork,
        **kwargs
    ) -> Placement:

        seed = kwargs.get("seed", None)

        # TODO: Add test that the number of qubits is big enough

        random.seed(seed)

        placement_dict: dict[int, int] = {}

        server_list = network.get_server_list()

        for vertex in dist_circ.vertex_list:

            server_full = True

            while server_full:

                random_server = random.choice(server_list)

                vertices_places = [
                    placed_v
                    for placed_v, server in placement_dict.items()
                    if server == random_server
                ]
                vertices_places = [
                    v for v in vertices_places
                    if dist_circ.vertex_circuit_map[v]['type'] == 'qubit'
                ]

                server_full = (len(vertices_places) == len(
                    network.server_qubits[random_server]))

            placement_dict[vertex] = random_server

        placement = Placement(placement_dict)
        assert placement.is_placement(dist_circ, network)

        return placement
