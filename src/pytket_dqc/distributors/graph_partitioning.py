from __future__ import annotations

import random
import kahypar as kahypar  # type:ignore
from pytket_dqc.distributors import Distributor, GainManager
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

        package_path = importlib_resources.files("pytket_dqc")
        default_ini = f"{package_path}/distributors/km1_kKaHyPar_sea20.ini"
        ini_path = kwargs.get("ini_path", default_ini)
        seed = kwargs.get("seed", None)
        num_rounds = kwargs.get("num_rounds", 1000)
        stop_parameter = kwargs.get("stop_parameter", 0.05)

        # First step is to call KaHyPar using the connectivity metric (i.e. no
        # knowledge about network topology other than server sizes)
        placement = self.initial_distribute(
            dist_circ, network, ini_path=ini_path, seed=seed
        )
        # Then, we refine the placement using label propagation. This will
        # also ensure that the servers do not exceed their qubit capacity
        placement = self.refine(
            placement,
            dist_circ,
            network,
            num_rounds=num_rounds,
            stop_parameter=stop_parameter,
        )

        return placement

    def refine(
        self,
        placement: Placement,
        dist_circ: DistributedCircuit,
        network: NISQNetwork,
        **kwargs,
    ) -> Placement:
        """The refinement algorithm proceeds in rounds. In each round, all of
        the vertices in the boundary are visited in random order and we
        we calculate the gain achieved by moving the vertex to other servers.
        The best move is applied, with ties broken randomly. If all possible
        moves have negative gains, the vertex is not moved; as such, the
        refinement algorithm cannot escape local optima. The justification
        is that we assume ``initial_distribute`` already left us close to
        a decent solution and we simply wish to refine it. This assumption
        is empirically justified when refinement is used in tandem with
        a coarsening approach.

        The algorithm continues until the proportion of vertices
        moved in a round (i.e. #moved / #boundary) is smaller than
        ``stop_parameter`` or the maximum ``num_rounds`` is reached.
        The resulting placement is guaranteed to be valid.

        This refinement algorithm is known as "label propagation" and it
        is discussed in https://arxiv.org/abs/1402.3281.

        :param placement: Initial placement.
        :type placement: Placement
        :param dist_circ: Circuit to distribute.
        :type dist_circ: DistributedCircuit
        :param network: Network onto which ``dist_circ`` should be placed.
        :type network: NISQNetwork

        :key seed: Seed for randomness. Default is None
        :key num_rounds: Max number of refinement rounds. Default is 1000.
        :key stop_parameter: Real number in [0,1]. If proportion of moves
            in a round is smaller than this number, do no more rounds. Default
            is 0.05.

        :raises Exception: Raised if there are more circuit qubits than
            physical qubits in the network

        :return: Placement of ``dist_circ`` onto ``network``.
        :rtype: Placement
        """

        num_rounds = kwargs.get("num_rounds", 1000)
        stop_parameter = kwargs.get("stop_parameter", 0.05)
        seed = kwargs.get("seed", None)
        if seed is not None:
            random.seed(seed)

        # We will use a ``GainManager`` to manage the calculation of gains
        # (and management of pre-computed values) in a transparent way
        gain_manager = GainManager(dist_circ, network, placement)

        round_id = 0
        proportion_moved: float = 1
        while round_id < num_rounds and proportion_moved > stop_parameter:
            boundary = dist_circ.get_boundary(placement)

            moves = 0
            for vertex in boundary:
                current_server = gain_manager.current_server(vertex)
                potential_servers = set(
                    gain_manager.current_server(v)
                    for v in dist_circ.vertex_neighbours[vertex]
                )
                potential_servers.add(gain_manager.current_server(vertex))

                best_server = None
                best_gain = 0
                for server in potential_servers:
                    # Servers that are not in ``potential_servers`` will always
                    # have the worst gain since they contain no neighbours
                    # of ``vertex``. As such, we  simply ignore them.

                    # If the move is not valid, skip it
                    if not gain_manager.is_move_valid(vertex, server):
                        continue

                    gain = gain_manager.gain(vertex, server)

                    if (
                        best_server is None
                        or gain > best_gain
                        or gain == best_gain
                        and random.choice([True, False])
                    ):
                        best_gain = gain
                        best_server = server

                # If no move within ``potential_servers`` is valid we move
                # ``vertex`` to a random server where it fits.
                # This is a last resort option and it is likely to never
                # occur.
                if best_server is None:
                    valid_servers = [
                        server
                        for server in network.get_server_list()
                        if gain_manager.is_move_valid(vertex, server)
                    ]
                    if not valid_servers:
                        raise Exception(
                            "Could not complete qubit allocation refinement. "
                            "More qubits in the circuit than in the network!"
                        )
                    best_server = random.choice(valid_servers)

                if best_server != current_server:
                    gain_manager.move(vertex, best_server)
                    moves += 1

            round_id += 1
            proportion_moved = (
                0 if len(boundary) == 0 else moves / len(boundary)
            )

        assert gain_manager.placement.is_valid(dist_circ, network)
        return gain_manager.placement

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

        ini_path = kwargs.get("ini_path")
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
