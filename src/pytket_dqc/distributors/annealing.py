from __future__ import annotations

from pytket_dqc.distributors import Distributor
from pytket_dqc.placement import Placement
from typing import TYPE_CHECKING
import random
from .ordered import Ordered

if TYPE_CHECKING:
    from pytket_dqc import DistributedCircuit
    from pytket_dqc.networks import NISQNetwork


class Annealing(Distributor):
    """Distributor taking a simulated annealing approach to quantum circuit
    distribution.
    """
    def __init__(self):
        """Initialisation function. No inputs required.
        """
        pass

    def distribute(
        self,
        dist_circ: DistributedCircuit,
        network: NISQNetwork,
        **kwargs
    ) -> Placement:
        """Distribute quantum circuit using simulated annealing approach.

        :param dist_circ: Circuit to distribute.
        :type dist_circ: DistributedCircuit
        :param network: Network onto which circuit is to be distributed.
        :type network: NISQNetwork
        :return: Placement of ``dist_circ`` onto ``network``
        :rtype: Placement

        :key seed: Seed for randomness. Default is None
        :key iterations: The number of iterations in the annealing procedure.
            Default is 10000.
        :initial_place_method: Distributor to use to find the initial
            placement. Default is Ordered.
        """

        iterations = kwargs.get("iterations", 10000)
        initial_distributor = kwargs.get("initial_place_method", Ordered())
        seed = kwargs.get("seed", None)

        random.seed(seed)

        # The annealing procedure requires an initial placement to work with.
        # An initial placement is arrived at here.
        placement = initial_distributor.distribute(dist_circ, network)
        cost = placement.cost(dist_circ, network)

        # TODO: Check that the initial placement does not have cost 0, and
        # that not all qubits are already in the same server etc.

        # For each step of the annealing process, try a slight altering of the
        # placement to see if it improves. Change to new placement if there is
        # an improvement. Change to new placement with small probability if
        # there is no improvement.
        for _ in range(iterations):

            if cost == 0:
                break

            # Choose a random vertex to move.
            vertex_to_move = random.choice(dist_circ.vertex_list)

            # find the server in which the chosen vertex resides.
            home_server = placement.placement[vertex_to_move]

            # Pick a random server to move to.
            possible_servers = network.get_server_list()
            possible_servers.remove(home_server)
            destination_server = random.choice(possible_servers)

            # Initialise new placement dictionary.
            swap_placement_dict = placement.placement.copy()

            # If the vertex to move corresponds to a qubit
            if dist_circ.vertex_circuit_map[vertex_to_move]['type'] == 'qubit':

                # List all qubit vertices
                destination_server_qubit_list = [
                    v for v in dist_circ.vertex_list
                    if (dist_circ.vertex_circuit_map[v]['type'] == 'qubit')
                ]

                # Remove qubits in the same server
                destination_server_qubit_list = [
                    v for v in destination_server_qubit_list
                    if (placement.placement[v] == destination_server)
                ]

                q_in_dest = len(destination_server_qubit_list)
                size_dest = len(network.server_qubits[destination_server])

                # If destination server is full, pick a random qubit in that
                # server and move it to the home server of the qubit
                # being moved.
                if q_in_dest == size_dest:
                    swap_vertex = random.choice(destination_server_qubit_list)
                    swap_placement_dict[swap_vertex] = home_server

            # Move qubit to new destination server.
            swap_placement_dict[vertex_to_move] = destination_server

            # Calculate cost of new placement.
            swap_placement = Placement(swap_placement_dict)
            swap_cost = swap_placement.cost(dist_circ, network)

            # If it is smaller than current cost, use placement as new
            # placement to compare against.
            if swap_cost < cost:
                placement = swap_placement
                cost = swap_cost

            # TODO: Annealing requires that the placement be swapped with some
            # small probability even if the cost does not improve, to avoid
            # local minima.

            assert placement.is_placement(dist_circ, network)

        return placement
