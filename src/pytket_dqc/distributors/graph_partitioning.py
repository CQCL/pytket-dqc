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
    package. This distributor is not guaranteed to return a valid placement
    as it will perform load balancing, which is to say an even placement of
    vertices onto servers. This distributor will ignore weighted hypergraphs
    and assume all hyperedges have weight 1. This distributor will ignore the
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
        stop_parameter = kwargs.get("stop_parameter",0.05)

        gain_manager = GainManager(dist_circ, placement)

        round_id = 0
        proportion_moved = 1
        while round_id < num_rounds and proportion_moved > stop_parameter:
            boundary = dist_circ.get_boundary(placement)

            moves = 0
            for vertex in boundary:
                neighbours = dist_circ.vertex_neighbours[vertex]
                neighbour_blocks = set([placement.placement[v] for v in neighbours])

                best_block = placement.placement[vertex]
                best_gain = 0
                for block in neighbour_blocks:
                    # TODO: Check, is this a valid move?

                    gain = gain(dist_circ, vertex, block)

                    if gain > best_gain or \
                       gain == best_gain and random.choice([True,False]):
                            best_gain = gain
                            best_block = block

                if best_block != placement.placement[vertex]:
                    placement.placement[vertex] = best_block
                    moves += 1

            round_id += 1
            proportion_moved = moves / len(boundary)

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
        """

        if not dist_circ.is_valid():
            raise Exception("This hypergraph is not valid.")

        hyperedge_indices, hyperedges = dist_circ.kahypar_hyperedges()

        num_hyperedges = len(hyperedge_indices) - 1
        num_vertices = len(list(set(hyperedges)))
        num_servers = len(network.get_server_list())

        hypergraph = kahypar.Hypergraph(
            num_vertices,
            num_hyperedges,
            hyperedge_indices,
            hyperedges,
            num_servers,
        )

        context = kahypar.Context()

        package_path = importlib_resources.files("pytket_dqc")
        default_ini = f"{package_path}/distributors/km1_kKaHyPar_sea20.ini"
        ini_path = kwargs.get("ini_path", default_ini)
        context.loadINIconfiguration(ini_path)

        context.setK(num_servers)
        context.setEpsilon(self.epsilon)
        context.suppressOutput(True)

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
        self.cache: dict[dict[int,int], int] = dict()

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



    # I should probably keep a cache of already computed costs for each hyperedge.
    # If I do so, I should probably create a class to manage this and put gain in it.

