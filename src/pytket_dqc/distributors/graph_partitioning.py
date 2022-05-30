from __future__ import annotations

import random
import kahypar as kahypar  # type:ignore
from pytket_dqc.distributors import Distributor
from pytket_dqc.placement import Placement
import importlib_resources


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pytket_dqc import DistributedCircuit
    from pytket_dqc.networks import NISQNetwork


class GraphPartitioning(Distributor):
    """Distribution technique, making use of existing tools for hypergraph
    partitioning available through the `kahypar <https://kahypar.org/>`_
    package. This distributor will ignore weighted hypergraphs and assume
    all hyperedges have weight 1. This distributor will ignore the
    connectivity of the NISQNetwork.
    """

    def __init__(self, epsilon: float = 0.03) -> None:
        """Initialisation function.

        :param epsilon: Load imbalance tolerance, defaults to 0.03
        :type epsilon: float, optional
        """
        self.epsilon = epsilon  # I think we can remove this

    def distribute(
        self, dist_circ: DistributedCircuit, network: NISQNetwork, **kwargs
    ) -> Placement:
        """Distribute ``dist_circ`` onto ``network``. The initial placement
        is found by KaHyPar using the connectivity metric, then it is
        refined to reduce the cost taking into account the network topology.

        :param dist_circ: Circuit to distribute.
        :type dist_circ: DistributedCircuit
        :param network: Network onto which ``dist_circ`` should be placed.
        :type network: NISQNetwork

        :key ini_path: Path to kahypar ini file.
        :key seed: Seed for randomness. Default is None
        :key num_rounds: Max number of refinement rounds. Default is 1000.
        :key stop_parameter: Real number in [0,1]. If proportion of moves
        in a round is smaller than this number, do no more rounds. Default
        is 0.05.

        :return: Placement of ``dist_circ`` onto ``network``.
        :rtype: Placement
        """

        seed = kwargs.get("seed", None)
        num_rounds = kwargs.get("num_rounds", 1000)
        stop_parameter = kwargs.get("stop_parameter", 0.05)

        # First step is to call KaHyPar using the connectivity metric (i.e. no
        # knowledge about network topology other than server sizes)
        placement = self.initial_distribute(dist_circ, network, seed=seed)

        # We will use a ``GainManager`` to manage the calculation of gains
        # (and management of pre-computed values) in a transparent way
        gain_manager = GainManager(dist_circ, network, placement)

        # The refinement algorithm proceeds in rounds. In each round, all of
        # the vertices in the boundary are visited in random order and we
        # we calculate the gain achieved by moving the vertex to each of its
        # neighbouring blocks. If all possibe moves of a given vertex have
        # negative gains, the vertex is not moved; otherwise the best move
        # is applied, with ties broken randomly.
        #
        # The algorithm continues until the proportion of vertices moved in
        # a round (i.e. #moved / #boundary) is smaller than ``stop_parameter``
        # or the maximum ``num_rounds`` is reached.
        #
        # This refinement algorithm is known as "label propagation" and it
        # was discussed in https://arxiv.org/abs/1402.3281.
        round_id = 0
        proportion_moved: float = 1
        while round_id < num_rounds and proportion_moved > stop_parameter:
            boundary = dist_circ.get_boundary(placement)

            moves = 0
            for vertex in boundary:
                neighbours = dist_circ.vertex_neighbours[vertex]
                neighbour_blocks = set(
                    [gain_manager.current_block(v) for v in neighbours]
                )

                best_block = gain_manager.current_block(vertex)
                best_gain = 0
                for block in neighbour_blocks:
                    # TODO: The current approach cannot move vertices to empty
                    # servers. This is probably a good thing since we are
                    # meant to assume that ``initial_distribute`` only left
                    # the least helpful servers empty, but edge cases may
                    # exist that make us reconsider.

                    # If the move is not valid, skip it
                    if not gain_manager.is_move_valid(vertex, block):
                        continue

                    gain = gain_manager.gain(vertex, block)

                    if (
                        gain > best_gain
                        or gain == best_gain
                        and random.choice([True, False])
                    ):
                        best_gain = gain
                        best_block = block

                if best_block != gain_manager.current_block(vertex):
                    gain_manager.move(vertex, best_block)
                    moves += 1

            round_id += 1
            proportion_moved = (
                0 if len(boundary) == 0 else moves / len(boundary)
            )

        return gain_manager.placement

    # TODO: dist_circ does not need to be a DistributedCircuit and could be a
    # Hypergraph. Is there a way of specifying this in the typing?
    def initial_distribute(
        self, dist_circ: DistributedCircuit, network: NISQNetwork, **kwargs
    ) -> Placement:
        """Distribute ``dist_circ`` onto ``network`` using graph partitioning
        tools available in `kahypar <https://kahypar.org/>`_ package. The
        placement returned is not taking into account network topology.
        However, it does take into account server sizes.

        :param dist_circ: Circuit to distribute.
        :type dist_circ: DistributedCircuit
        :param network: Network onto which ``dist_circ`` should be placed.
        :type network: NISQNetwork

        :key ini_path: Path to kahypar ini file.
        :key seed: Seed for randomness. Default is None

        :return: Placement of ``dist_circ`` onto ``network``.
        :rtype: Placement
        """

        if not dist_circ.is_valid():
            raise Exception("This hypergraph is not valid.")

        seed = kwargs.get("seed", None)

        hyperedge_indices, hyperedges = dist_circ.kahypar_hyperedges()

        num_hyperedges = len(hyperedge_indices) - 1
        num_vertices = len(list(set(hyperedges)))
        server_list = network.get_server_list()
        num_servers = len(server_list)
        server_sizes = [len(network.server_qubits[s]) for s in server_list]
        # For now, all hyperedges are assumed to have the same weight
        hyperedge_weights = [1 for i in range(0, num_hyperedges)]
        # Qubit vertices are given weight 1, gate vertices are given weight 0
        num_qubits = len(dist_circ.circuit.qubits)
        vertex_weights = [1 for i in range(0, num_qubits)] + [
            0 for i in range(num_qubits, num_vertices)
        ]
        # TODO: the weight assignment to vertices assumes that the index of the
        # qubit vertices range from 0 to ``num_qubits``, and the rest of them
        # correspond to gates. This is currently guaranteed by definition of
        # ``from_circuit()``, but we might want to make this more robust.

        hypergraph = kahypar.Hypergraph(
            num_vertices,
            num_hyperedges,
            hyperedge_indices,
            hyperedges,
            num_servers,
            hyperedge_weights,
            vertex_weights,
        )

        context = kahypar.Context()

        package_path = importlib_resources.files("pytket_dqc")
        default_ini = f"{package_path}/distributors/km1_kKaHyPar_sea20.ini"
        ini_path = kwargs.get("ini_path", default_ini)
        context.loadINIconfiguration(ini_path)

        context.setK(num_servers)
        context.setEpsilon(self.epsilon)
        context.setCustomTargetBlockWeights(server_sizes)
        context.suppressOutput(True)
        if seed is not None:
            context.setSeed(seed)

        kahypar.partition(hypergraph, context)

        partition_list = [
            hypergraph.blockID(i) for i in range(hypergraph.numNodes())
        ]

        placement_dict = {i: server for i, server in enumerate(partition_list)}

        return Placement(placement_dict)


class GainManager:
    """Instances of this class are used to manage pre-computed values of the
    gain of a move, since it is likely that the same value will be used
    multiple times and computing it requires solving a minimum spanning tree
    problem which takes non-negligible computation time.

    :param dist_circ: The circuit to be distributed, carries hypergraph info
    :type dist_circ: DistributedCircuit
    :param network: The network topology that the circuit must be mapped to
    :type network: NISQNetwork
    :param placement: The current placement
    :type placement: Placement
    :param occupancy: Maps servers to its current number of qubit vertices
    :type occupancy: dict[int, int]
    :param cache: A dictionary of sets of blocks to their communication cost
    :type cache: dict[frozenset[int], int]
    """

    # TODO: Might be worth it to give a max size of the cache to avoid
    # storing too much...

    def __init__(
        self,
        dist_circ: DistributedCircuit,
        network: NISQNetwork,
        placement: Placement,
    ):
        self.dist_circ: DistributedCircuit = dist_circ
        self.network: NISQNetwork = network
        self.placement: Placement = placement
        self.cache: dict[frozenset[int], int] = dict()
        self.occupancy: dict[int, int] = dict()

        for vertex, server in placement.placement.items():
            if dist_circ.is_qubit_vertex(vertex):
                if server not in self.occupancy.keys():
                    self.occupancy[server] = 0
                self.occupancy[server] += 1

    def is_move_valid(self, vertex: int, server: int) -> bool:
        """ The move is only invalid when ``vertex`` is a qubit vertex and
        ``server`` is at its maximum occupancy.
        """
        return not (
            self.dist_circ.is_qubit_vertex(vertex)
            and self.occupancy[server]
            >= len(self.network.server_qubits[server])
        )

    def gain(self, vertex: int, new_block: int) -> int:
        """Compute the gain of moving ``vertex`` to ``new_block``. Instead
        of calculating the cost of the whole hypergraph using the new
        placement, we simply compare the previous cost of all hyperedges
        incident to ``vertex`` and substract their new cost. Moreover, if
        these values are available in the cache they are used; otherwise,
        the cache is updated.

        The formula to calculate the gain comes from the gain function
        used in KaHyPar for the connectivity metric, as in the dissertation
        (https://publikationen.bibliothek.kit.edu/1000105953), where weights
        are calculated using spanning trees over ``network``. This follows
        suggestions from Tobias Heuer.
        """
        current_block = self.placement.placement[vertex]
        if current_block == new_block:
            return 0

        gain = 0
        loss = 0
        for hyperedge in self.dist_circ.hyperedge_dict[vertex]:
            # Set of connected blocks omitting that of the vertex being moved
            connected_blocks = [
                self.placement.placement[v]
                for v in hyperedge.vertices
                if v != vertex
            ]

            current_block_pins = len(
                [b for b in connected_blocks if b == current_block]
            )
            new_block_pins = len(
                [b for b in connected_blocks if b == new_block]
            )

            # The cost of hyperedge will only be decreased by the move if
            # ``vertex`` is was the last member of ``hyperedge`` in
            # ``current_block``
            if current_block_pins == 0:
                cache_key = frozenset(connected_blocks + [current_block])
                if cache_key not in self.cache.keys():
                    self.cache[
                        cache_key
                    ] = 0  # hyperedge_cost(hyperedge, placement)
                gain += self.cache[cache_key]

            # The cost of hyperedge will only be increased by the move if
            # no vertices from ``hyperedge`` were in ``new_block`` prior
            # to the move
            if new_block_pins == 0:
                cache_key = frozenset(connected_blocks + [new_block])
                if cache_key not in self.cache.keys():
                    self.cache[
                        cache_key
                    ] = 0  # hyperedge_cost(hyperedge, new_placement)
                loss += self.cache[cache_key]

        return gain - loss

    def move(self, vertex: int, block: int):
        """Moves ``vertex`` to ``block``, updating ``placement`` and
        ``occupancy`` accordingly.
        """
        self.occupancy[block] += 1
        self.occupancy[self.placement.placement[vertex]] -= 1
        self.placement.placement[vertex] = block

    def current_block(self, vertex: int):
        """Just an alias to make code clearer.
        """
        return self.placement.placement[vertex]
