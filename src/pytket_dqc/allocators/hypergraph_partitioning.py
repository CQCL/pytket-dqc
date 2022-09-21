from __future__ import annotations

import random
import kahypar as kahypar  # type:ignore
from pytket_dqc.allocators import Allocator, GainManager
from pytket_dqc.placement import Placement
from pytket_dqc.circuits.distribution import Distribution
import importlib_resources
from pytket import Circuit


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pytket_dqc import HypergraphCircuit
    from pytket_dqc.networks import NISQNetwork


class HypergraphPartitioning(Allocator):
    """Distribution technique, making use of existing tools for hypergraph
    partitioning available through the `kahypar <https://kahypar.org/>`_
    package. This allocator will ignore weighted hypergraphs and assume
    all hyperedges have weight 1. This allocator will ignore the
    connectivity of the NISQNetwork.
    """

    def allocate(
        self, dist_circ: HypergraphCircuit, network: NISQNetwork, **kwargs
    ) -> Distribution:
        """Distribute ``dist_circ`` onto ``network``. The initial distribution
        is found by KaHyPar using the connectivity metric, then it is
        refined to reduce the cost taking into account the network topology.

        :param dist_circ: Circuit to distribute.
        :type dist_circ: HypergraphCircuit
        :param network: Network onto which ``dist_circ`` should be placed.
        :type network: NISQNetwork

        :key ini_path: Path to kahypar ini file.
        :key seed: Seed for randomness. Default is None
        :key num_rounds: Max number of refinement rounds. Default is 1000.
        :key stop_parameter: Real number in [0,1]. If proportion of moves
            in a round is smaller than this number, do no more rounds. Default
            is 0.05.
        :key cache_limit: The maximum size of the set of servers whose cost is
            stored in cache; see GainManager. Default value is 5.

        :return: Distribution of ``dist_circ`` onto ``network``.
        :rtype: Distribution
        """

        if not network.can_implement(dist_circ):
            raise Exception(
                "This circuit cannot be implemented on this network."
            )

        package_path = importlib_resources.files("pytket_dqc")
        default_ini = f"{package_path}/allocators/km1_kKaHyPar_sea20.ini"
        ini_path = kwargs.get("ini_path", default_ini)
        seed = kwargs.get("seed", None)
        num_rounds = kwargs.get("num_rounds", 1000)
        stop_parameter = kwargs.get("stop_parameter", 0.05)
        cache_limit = kwargs.get("cache_limit", None)

        # First step is to call KaHyPar using the connectivity metric (i.e. no
        # knowledge about network topology other than server sizes)
        placement = self.initial_distribute(
            dist_circ, network, ini_path, seed=seed
        )

        initial_distribution = Distribution(
            dist_circ, dist_circ, placement, network
        )

        # Then, we refine the placement using label propagation. This will
        # also ensure that the servers do not exceed their qubit capacity
        distribution = self.refine(
            initial_distribution,
            seed=seed,
            num_rounds=num_rounds,
            stop_parameter=stop_parameter,
            cache_limit=cache_limit,
        )

        assert distribution.is_valid()
        return distribution

    def refine(
        self, initial_distribution: Distribution, **kwargs,
    ) -> Distribution:
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

        :param initial_distribution: Distribution to refine.
        :type distribution: Distribution

        :key seed: Seed for randomness. Default is None
        :key num_rounds: Max number of refinement rounds. Default is 1000.
        :key stop_parameter: Real number in [0,1]. If proportion of moves
            in a round is smaller than this number, do no more rounds. Default
            is 0.05.
        :key cache_limit: The maximum size of the set of servers whose cost is
            stored in cache; see GainManager. Default value is 5.

        :return: Distribution where the placement updated.
        :rtype: Distribution
        """

        num_rounds = kwargs.get("num_rounds", 1000)
        stop_parameter = kwargs.get("stop_parameter", 0.05)
        seed = kwargs.get("seed", None)
        if seed is not None:
            random.seed(seed)
        cache_limit = kwargs.get("cache_limit", None)

        # We will use a ``GainManager`` to manage the calculation of gains
        # (and management of pre-computed values) in a transparent way
        gain_manager = GainManager(initial_distribution)
        if cache_limit is not None:
            gain_manager.set_max_key_size(cache_limit)

        # Since KaHyPar does not guarantee that the requirement on server
        # capacity will be satisfied, we enforce this ourselves.
        # However, it usually does satisfy the requirement and the following
        # code often does nothing or it moves very few vertices.
        for vertex in gain_manager.qubit_vertices:
            # If moving in place is not valid then the current server is full
            if not gain_manager.is_move_valid(
                vertex, gain_manager.current_server(vertex)
            ):
                # Then, find the first server where the move would be valid
                for server in gain_manager.occupancy.keys():
                    if gain_manager.is_move_valid(vertex, server):
                        # Move ``vertex`` to a server with free spaces and
                        # return control to the outer loop
                        gain_manager.move(vertex, server)
                        break
        # Notice that the moves have been arbitrary, i.e. we have not
        # calculated gains. This is fine since the vertices we moved will
        # likely be boundary vertices; the following rounds of the
        # refinement algorithm will move them around to optimise gains.
        # At the end of the previous subroutine, no server should be
        # overpopulated.
        assert gain_manager.distribution.is_valid()

        round_id = 0
        proportion_moved: float = 1
        dist_circ = gain_manager.distribution.circuit
        placement = gain_manager.distribution.placement
        while round_id < num_rounds and proportion_moved > stop_parameter:
            active_vertices = dist_circ.get_boundary(placement)

            moves = 0
            for vertex in active_vertices:
                current_server = gain_manager.current_server(vertex)
                # We only consider moving ``vertex`` to a server that has
                # a neighbour vertex allocated to it
                potential_servers = set(
                    gain_manager.current_server(v)
                    for v in dist_circ.vertex_neighbours[vertex]
                )
                # We explicitly add the current server to the set
                # i.e. a potentially valid move is doing no move at all
                potential_servers.add(gain_manager.current_server(vertex))

                best_server = None
                best_gain = float("-inf")
                best_best_swap = None
                for server in potential_servers:
                    # Servers that are not in ``potential_servers`` will always
                    # have the worst gain since they contain no neighbours
                    # of ``vertex``. As such, we  simply ignore them.

                    gain = gain_manager.gain(vertex, server)

                    # If the move is not valid (i.e. the server is full) we
                    # find the best vertex in ``server`` to swap this one with
                    best_swap_vertex = None
                    if not gain_manager.is_move_valid(vertex, server):
                        # The only vertices we can swap with are qubit ones
                        # so that the occupancy of the server is maintained
                        vs = placement.get_vertices_in(
                            server
                        )
                        valid_swaps = [
                            vertex
                            for vertex in vs
                            if dist_circ.is_qubit_vertex(vertex)
                        ]

                        # To obtain the gain accurately, we move ``vertex`` to
                        # ``server`` first and move it back at the end. This is
                        # possible because ``move`` is an unsafe function, i.e.
                        # it does not require that the move is valid.
                        gain_manager.move(vertex, server)

                        best_swap_gain = float("-inf")
                        for swap_vertex in valid_swaps:
                            swap_gain = gain_manager.gain(
                                swap_vertex, current_server
                            )

                            if (
                                best_swap_vertex is None
                                or swap_gain > best_swap_gain
                                or swap_gain == best_swap_gain
                                and random.choice([True, False])
                            ):
                                best_swap_gain = swap_gain
                                best_swap_vertex = swap_vertex
                        # Restore ``vertex`` to its original server.
                        gain_manager.move(vertex, current_server)

                        # Since no server has capacity 0, we should always
                        # find a vertex to swap with
                        assert best_swap_vertex is not None
                        # The gain of this swap is the sum of the gains of
                        # both moves
                        gain = gain + int(best_swap_gain)

                    if (
                        best_server is None
                        or gain > best_gain
                        or gain == best_gain
                        and random.choice([True, False])
                    ):
                        best_gain = gain
                        best_server = server
                        # ``best_swap_vertex`` contains either None (if the
                        # move was valid) or the best vertex to swap with for
                        # this particular ``server``. But, since this variable
                        # will be initialised once for each server we attempt
                        # to move to, we need to store the best swap of the
                        # best server somewhere: that is ``best_best_swap``
                        best_best_swap = best_swap_vertex

                # Since ``potential_servers`` includes at least the
                # ``current_server``, there is always at least one server
                # to choose from
                assert best_server is not None

                if best_server != current_server:
                    gain_manager.move(vertex, best_server)
                    if best_best_swap is not None:
                        # This means that the move was not valid, so we need
                        # to swap to make it valid
                        gain_manager.move(best_best_swap, current_server)
                    # Either if we swap or we don't, we count it as one move
                    # since this is meant to count 'rounds with change' rather
                    # than literal moves
                    moves += 1

            round_id += 1
            proportion_moved = (
                moves / len(active_vertices) if active_vertices else 0
            )

        assert gain_manager.distribution.is_valid()
        return gain_manager.distribution

    def initial_distribute(
        self,
        dist_circ: HypergraphCircuit,
        network: NISQNetwork,
        ini_path: str,
        **kwargs,
    ) -> Placement:
        """Distribute ``dist_circ`` onto ``network`` using graph partitioning
        tools available in `kahypar <https://kahypar.org/>`_ package. The
        placement returned is not taking into account network topology.
        However, it does take into account server sizes.

        :param dist_circ: Circuit to distribute.
        :type dist_circ: HypergraphCircuit
        :param network: Network onto which ``dist_circ`` should be placed.
        :type network: NISQNetwork
        :param ini_path: Path to kahypar ini file.
        :type ini_path: str

        :key seed: Seed for randomness. Default is None

        :return: Placement of ``dist_circ`` onto ``network``.
        :rtype: Placement
        """

        if not dist_circ.is_valid():
            raise Exception("This hypergraph is not valid.")

        seed = kwargs.get("seed", None)

        # This should only arise if the circuit is completely empty.
        if len(dist_circ.hyperedge_list) == 0:
            assert dist_circ.circuit == Circuit()
            return Placement(dict())
        else:
            hyperedge_indices, hyperedges = dist_circ.kahypar_hyperedges()

            num_hyperedges = len(hyperedge_indices) - 1
            num_vertices = len(list(set(hyperedges)))
            server_list = network.get_server_list()
            num_servers = len(server_list)
            server_sizes = [len(network.server_qubits[s]) for s in server_list]
            # For now, all hyperedges are assumed to have the same weight
            hyperedge_weights = [1 for i in range(0, num_hyperedges)]
            # Qubit vertices are given weight 1, gate vertices are given
            # weight 0
            num_qubits = len(dist_circ.circuit.qubits)
            vertex_weights = [1 for i in range(0, num_qubits)] + [
                0 for i in range(num_qubits, num_vertices)
            ]
            # TODO: the weight assignment to vertices assumes that the index
            # of the qubit vertices range from 0 to `num_qubits`, and the
            # rest of them correspond to gates. This is currently guaranteed
            # by construction i.e. method `from_circuit()`; we might want
            # to make this more robust.
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
            default_ini = f"{package_path}/allocators/km1_kKaHyPar_sea20.ini"
            ini_path = kwargs.get("ini_path", default_ini)
            context.loadINIconfiguration(ini_path)

            context.setK(num_servers)
            context.setCustomTargetBlockWeights(server_sizes)
            context.suppressOutput(True)
            if seed is not None:
                context.setSeed(seed)

            kahypar.partition(hypergraph, context)

            partition_list = [
                hypergraph.blockID(i) for i in range(hypergraph.numNodes())
            ]

            placement_dict = {
                i: server for i, server in enumerate(partition_list)
            }
            placement = Placement(placement_dict)

        return placement
