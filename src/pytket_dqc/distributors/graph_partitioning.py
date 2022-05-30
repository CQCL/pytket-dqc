from __future__ import annotations

import random
import kahypar as kahypar  # type:ignore
from pytket_dqc.distributors import Distributor
from pytket_dqc.placement import Placement
import importlib_resources


from typing import TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    from pytket_dqc import DistributedCircuit, Hypergraph
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
        """Distribute ``dist_circ`` onto ``network`` using graph partitioning
        tools available in `kahypar <https://kahypar.org/>`_ package. This
        may not return a valid placement.

        :param dist_circ: Circuit to distribute.
        :type dist_circ: DistributedCircuit
        :param network: Network onto which ``dist_circ`` should be placed.
        :type network: NISQNetwork

        :key ini_path: Path to kahypar ini file.

        :return: Placement of ``dist_circ`` onto ``network``.
        :rtype: Placement

        :key num_rounds: Max number of refinement rounds. Default is 1000.
        :key stop_parameter: Real number in [0,1]. If proportion of moves
        in a round is smaller than this number, do no more rounds. Default
        is 0.05.
        """

        # First step is to call KaHyPar using the connectivity metric (i.e. no
        # knowledge about network topology other than server sizes)
        placement = self.initial_distribute(
            dist_circ, network
        )  # TODO: Add the seed

        num_rounds = kwargs.get("num_rounds", 1000)
        stop_parameter = kwargs.get("stop_parameter", 0.05)

        gain_manager = GainManager(dist_circ, placement)

        round_id = 0
        proportion_moved: float = 1
        while round_id < num_rounds and proportion_moved > stop_parameter:
            boundary = dist_circ.get_boundary(placement)

            moves = 0
            for vertex in boundary:
                neighbours = dist_circ.vertex_neighbours[vertex]
                neighbour_blocks = set(
                    [placement.placement[v] for v in neighbours]
                )

                best_block = placement.placement[vertex]
                best_gain = 0
                for block in neighbour_blocks:
                    # TODO: Check, is this a valid move?

                    gain = gain_manager.gain(vertex, block)

                    if (
                        gain > best_gain
                        or gain == best_gain
                        and random.choice([True, False])
                    ):
                        best_gain = gain
                        best_block = block

                if best_block != placement.placement[vertex]:
                    placement.placement[vertex] = best_block
                    moves += 1

            round_id += 1
            proportion_moved = (
                0 if len(boundary) == 0 else moves / len(boundary)
            )

        return placement

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

        :return: Placement of ``dist_circ`` onto ``network``.
        :rtype: Placement

        :key seed: Seed for randomness. Default is None
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
        # qubit vertices range from 0 to `num_qubits`, and the rest of them
        # correspond to gates. This is currently guaranteed by construction
        # i.e. method `from_circuit()`; we might want to make this more robust.

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
    """
    TODO
    """

    def __init__(self, hypergraph: Hypergraph, placement: Placement):
        """
        TODO
        """
        self.hypergraph: Hypergraph = hypergraph
        self.placement: Placement = placement
        self.cache: dict[frozenset[Tuple[int, int]], int] = dict()

    def update_placement(self, placement: Placement):
        """
        NOTE: since placements are changed on-site, the placement pointer does not change and we don't need to call this ever.
        """
        self.placement = placement

    def clear_cache(self):
        """
        TODO
        """
        self.cache = dict()

    def gain(self, vertex: int, new_block: int) -> int:
        """
        TODO
        """
        current_block = self.placement.placement[vertex]

        gain = 0
        loss = 0
        for hyperedge in self.hypergraph.hyperedge_dict[vertex]:
            # Edge placement pairs without the vertex being moved
            edge_placement = [
                (v, self.placement.placement[v])
                for v in hyperedge.vertices
                if v != vertex
            ]

            current_block_pins = len(
                [b for b in edge_placement if b == current_block]
            )
            new_block_pins = len([b for b in edge_placement if b == new_block])

            # The cost of hyperedge will only be decreased by the move if
            # `vertex` is was the last member of `hyperedge` in `current_block`
            if current_block_pins == 0:
                cache_key = frozenset(
                    edge_placement + [(vertex, current_block)]
                )
                if cache_key not in self.cache.keys():
                    self.cache[
                        cache_key
                    ] = 0  # hyperedge_cost(hyperedge, placement)
                gain += self.cache[cache_key]

            # The cost of hyperedge will only be increased by the move if
            # no vertices from `hyperedge` were in `new_block` prior to
            # the move
            if new_block_pins == 0:
                cache_key = frozenset(edge_placement + [(vertex, new_block)])
                if cache_key not in self.cache.keys():
                    self.cache[
                        cache_key
                    ] = 0  # hyperedge_cost(hyperedge, new_placement)
                loss += self.cache[cache_key]

        return gain - loss
