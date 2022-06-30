from __future__ import annotations

from pytket_dqc.distributors import Distributor
from pytket_dqc.placement import Placement
from typing import TYPE_CHECKING
import random

if TYPE_CHECKING:
    from pytket_dqc import DistributedCircuit
    from pytket_dqc.networks import NISQNetwork


class Random(Distributor):
    """Distribute hypergraph vertices onto servers at random. The resulting
    placement is valid, which is to say vertices will not be placed on servers
    once that are full.
    """
    def __init__(self) -> None:
        pass

    def distribute(
        self,
        dist_circ: DistributedCircuit,
        network: NISQNetwork,
        **kwargs
    ) -> Placement:
        """Distribute ``dist_circ`` onto ``network`` by randomly placing
        vertices onto servers. Qubit vertices are placed onto servers
        until the server is full. Gate vertices are placed on servers at random
        without restriction.

        :param dist_circ: Circuit to distribute.
        :type dist_circ: DistributedCircuit
        :param network: Network onto which ``dist_circ`` should be distributed.
        :type network: NISQNetwork
        :return: Placement of ``dist_circ`` onto ``network``.
        :rtype: Placement
        """

        if not network.can_implement(dist_circ):
            raise Exception(
                "This circuit cannot be implemented on this network."
                )

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
                    v for v in vertices_places
                    if dist_circ.vertex_circuit_map[v]['type'] == 'qubit'
                ]

                # Check if server is full
                server_full = (len(vertices_places) == len(
                    network.server_qubits[random_server]))

            placement_dict[vertex] = random_server

        placement = Placement(placement_dict)
        assert placement.is_valid(dist_circ, network)

        return placement
