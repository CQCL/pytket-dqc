from __future__ import annotations

from pytket_dqc.distributors import Distributor
from pytket_dqc.placement import Placement
from typing import TYPE_CHECKING
import random
from .random import Random
import math

if TYPE_CHECKING:
    from pytket_dqc import DistributedCircuit
    from pytket_dqc.networks import NISQNetwork


def acceptance_criterion(
    new: int,
    current: int,
    iteration: int,
    initial_temperature: float = 1
):
    """Acceptance criterion, to be used during simulated annealing

    :param new: New value of objective function.
    :type new: int
    :param current: Current value of objective function.
    :type current: int
    :param iteration: Current iteration.
    :type iteration: int
    :param initial_temperature: Initial temperatire, defaults to 1.
    :type initial_temperature: float, optional
    :return: acceptance criterion.
    :rtype: [type]
    """

    temperature = initial_temperature / (iteration + 1)
    return math.exp(-(new - current)/temperature)


class Annealing(Distributor):
    """Distributor taking a simulated annealing approach to quantum circuit
    distribution.
    """

    def __init__(self) -> None:
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
        :return: Placement of ``dist_circ`` onto ``network``.
        :rtype: Placement

        :key seed: Seed for randomness. Default is None
        :key iterations: The number of iterations in the annealing procedure.
            Default is 10000.
        :initial_place_method: Distributor to use to find the initial
            placement. Default is Random.
        """

        iterations = kwargs.get("iterations", 10000)
        initial_distributor = kwargs.get("initial_place_method", Random())
        seed = kwargs.get("seed", None)

        random.seed(seed)

        # The annealing procedure requires an initial placement to work with.
        # An initial placement is arrived at here.
        current_placement = initial_distributor.distribute(dist_circ, network)
        current_cost = current_placement.cost(dist_circ, network)

        # TODO: Check that the initial placement does not have cost 0, and
        # that not all qubits are already in the same server etc.

        # For each step of the annealing process, try a slight altering of the
        # placement to see if it improves. Change to new placement if there is
        # an improvement. Change to new placement with small probability if
        # there is no improvement.
        for i in range(iterations):

            if current_cost == 0:
                break

            # Choose a random vertex to move.
            vertex_to_move = random.choice(dist_circ.vertex_list)

            # find the server in which the chosen vertex resides.
            home_server = current_placement.placement[vertex_to_move]

            # Pick a random server to move to.
            possible_servers = network.get_server_list()
            possible_servers.remove(home_server)
            destination_server = random.choice(possible_servers)

            # Initialise new placement dictionary.
            swap_placement_dict = current_placement.placement.copy()

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
                    if (current_placement.placement[v] == destination_server)
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

            acceptance_prob = acceptance_criterion(swap_cost, current_cost, i)

            # If acceptance probability is higher than random number then
            # ten accept new placement. Note that new placement is always
            # accepted if it is better than the old one.
            # TODO: Should new placement be accepted if the cost
            # does not change?
            if acceptance_prob > random.uniform(0, 1):
                current_placement = swap_placement
                current_cost = swap_cost

            assert current_placement.is_placement(dist_circ, network)

        return current_placement
