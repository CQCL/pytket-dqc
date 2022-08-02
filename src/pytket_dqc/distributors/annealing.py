from __future__ import annotations

from pytket_dqc.distributors import Distributor, GainManager
from pytket_dqc.placement import Placement
from typing import TYPE_CHECKING
import random
from .random import Random
import math

if TYPE_CHECKING:
    from pytket_dqc import DistributedCircuit
    from pytket_dqc.networks import NISQNetwork


# Credit to
# machinelearningmastery.com/simulated-annealing-from-scratch-in-python/
# for details of this implementation.


def acceptance_criterion(
    gain: int, iteration: int, initial_temperature: float = 1
):
    """Acceptance criterion, to be used during simulated annealing.
    If ``gain`` is positive (improvement) the output will be greater than 1.
    If ``gain`` is negative the output will be a probability from 0 to 1.

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
    try:
        acceptance = math.exp(gain / temperature)
    except OverflowError:
        acceptance = float("inf")
    return acceptance


class Annealing(Distributor):
    """Distributor taking a simulated annealing approach to quantum circuit
    distribution.
    """

    def __init__(self) -> None:
        pass

    def distribute(
        self, dist_circ: DistributedCircuit, network: NISQNetwork, **kwargs
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
        :key initial_place_method: Distributor to use to find the initial
            placement. Default is Random.
        :key cache_limit: The maximum size of the set of servers whose cost is
            stored in cache; see GainManager. Default value is 5.
        :key brute_force_steiner: If True, the Steiner tree subproblem is
            solved optimally using a brute force approach, otherwise it is
            solved using a poly-time 2-approximate algorithm. Default True.
        """

        if not network.can_implement(dist_circ):
            raise Exception(
                "This circuit cannot be implemented on this network."
            )

        iterations = kwargs.get("iterations", 10000)
        initial_distributor = kwargs.get("initial_place_method", Random())
        seed = kwargs.get("seed", None)
        cache_limit = kwargs.get("cache_limit", None)
        brute_force_steiner = kwargs.get("brute_force_steiner", True)

        random.seed(seed)

        qubit_vertices = frozenset(
            [v for v in dist_circ.vertex_list if dist_circ.is_qubit_vertex(v)]
        )

        # The annealing procedure requires an initial placement to work with.
        # An initial placement is arrived at here.
        initial_placement = initial_distributor.distribute(dist_circ, network)

        # TODO: Check that the initial placement does not have cost 0, and
        # that not all qubits are already in the same server etc.

        # We will use a ``GainManager`` to manage the calculation of gains
        # (and management of pre-computed values) in a transparent way
        gain_manager = GainManager(
            dist_circ, qubit_vertices, network, initial_placement
        )
        if cache_limit is not None:
            gain_manager.set_max_key_size(cache_limit)

        # For each step of the annealing process, try a slight altering of the
        # placement to see if it improves. Change to new placement if there is
        # an improvement. Change to new placement with small probability if
        # there is no improvement.
        for i in range(iterations):

            # Choose a random vertex to move.
            vertex_to_move = random.choice(dist_circ.vertex_list)

            # Find the server in which the chosen vertex resides.
            home_server = gain_manager.current_server(vertex_to_move)

            # Pick a random server to move to.
            possible_servers = network.get_server_list()
            possible_servers.remove(home_server)
            destination_server = random.choice(possible_servers)

            # If the vertex to move corresponds to a qubit
            swap_vertex = None
            if dist_circ.vertex_circuit_map[vertex_to_move]["type"] == "qubit":

                # List all qubit vertices
                destination_server_qubit_list = [
                    v
                    for v in dist_circ.vertex_list
                    if (dist_circ.vertex_circuit_map[v]["type"] == "qubit")
                ]

                # Gather qubits in ``destination_server``
                destination_server_qubit_list = [
                    v
                    for v in destination_server_qubit_list
                    if (gain_manager.current_server(v) == destination_server)
                ]

                q_in_dest = len(destination_server_qubit_list)
                size_dest = len(network.server_qubits[destination_server])

                # If destination server is full, pick a random qubit in that
                # server and move it to the home server of the qubit
                # being moved.
                if q_in_dest == size_dest:
                    swap_vertex = random.choice(destination_server_qubit_list)

            # Calculate gain
            gain = gain_manager.gain(
                vertex_to_move, destination_server, brute_force_steiner
            )
            if swap_vertex is not None:
                # In order to accurately calculate the gain of moving
                # ``swap_vertex`` we need to move ``vertex_to_move``
                gain_manager.move(vertex_to_move, destination_server)
                gain += gain_manager.gain(
                    swap_vertex, home_server, brute_force_steiner
                )
                # Restore ``vertex_to_move`` to its original placement
                gain_manager.move(vertex_to_move, home_server)

            acceptance_prob = acceptance_criterion(gain, i)

            # If acceptance probability is higher than random number then
            # ten accept new placement. Note that new placement is always
            # accepted if it is better than the old one.
            # TODO: Should new placement be accepted if the cost
            # does not change?
            if acceptance_prob > random.uniform(0, 1):
                gain_manager.move(vertex_to_move, destination_server)
                if swap_vertex is not None:
                    gain_manager.move(swap_vertex, home_server)

        assert gain_manager.placement.is_valid(dist_circ, network)
        return gain_manager.placement
