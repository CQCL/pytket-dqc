from __future__ import annotations

import random
import kahypar as kahypar  # type:ignore
from pytket_dqc.distributors import Distributor, GainManager
from pytket_dqc.placement import Placement
import importlib_resources
from pytket import Circuit


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pytket_dqc import DistributedCircuit
    from pytket_dqc.networks import NISQNetwork
    from pytket_dqc.circuits import CoarseHyp


class GraphPartitioning(Distributor):
    """Distribution technique, making use of existing tools for hypergraph
    partitioning available through the `kahypar <https://kahypar.org/>`_
    package. This distributor will ignore weighted hypergraphs and assume
    all hyperedges have weight 1. This distributor will ignore the
    connectivity of the NISQNetwork.
    """

    def distribute(self, dist_circ: DistributedCircuit, network: NISQNetwork, **kwargs) -> Placement:
        """Distribute ``hypergraph`` onto ``network``. First, there is a
        coarsening stage after which the initial placement is computed
        using KaHyPar with the connectivity metric, then it is
        refined to reduce the cost taking into account the network topology.

        :param dist_circ: Circuit to distribute
        :type dist_circ: DistributedCircuit
        :param network: Network onto which the circuit should be placed
        :type network: NISQNetwork

        :key ini_path: Path to kahypar ini file
        :key seed: Seed for the random number generator
        :key num_rounds: Max number of refinement rounds. Default is 1000.
        :key stop_parameter: Real number in [0,1]. If proportion of moves
            in a round is smaller than this number, do no more rounds. Default
            is 0.05.
        :key cache_limit: The maximum size of the set of servers whose cost is
            stored in cache; see GainManager. Default value is 5.

        :return: Placement of ``hypergraph`` onto ``network``.
        :rtype: Placement
        """
        package_path = importlib_resources.files("pytket_dqc")
        default_ini = f"{package_path}/distributors/km1_kKaHyPar_sea20.ini"
        ini_path = kwargs.get("ini_path", default_ini)
        seed = kwargs.get("seed", None)
        random.seed(seed)
        num_rounds = kwargs.get("num_rounds", 1000)
        stop_parameter = kwargs.get("stop_parameter", 0.05)
        cache_limit = kwargs.get("cache_limit", 5)

        if not dist_circ.is_valid():
            raise Exception("This hypergraph is not valid.")

        # This should only arise if the circuit is completely empty.
        if (len(dist_circ.hyperedge_list) == 0):
            assert dist_circ.circuit == Circuit()
            return Placement(dict())
        else:
            # We will use a ``GainManager`` to manage the calculation of gains
            # (and management of pre-computed values) in a transparent way
            qubit_vertices = [
                v for v in dist_circ.vertex_list if dist_circ.is_qubit_vertex(v)
            ]
            gm = GainManager(dist_circ, qubit_vertices, network, cache_limit)

            # First step is to coarsen the hypergraph until all vertices are qubit
            # vertices
            for i in range(len(gm.hypergraph.vertex_list) - len(gm.hypergraph.qubit_vertices)):
                self.coarsen_once(gm)
            # The set of non-hidden vertices should match the set of qubit vertices
            assert set(gm.hypergraph.vertex_list) - gm.hypergraph.hidden_vertices == gm.hypergraph.qubit_vertices

            # Then, we call KaHyPar using the connectivity metric (i.e. no
            # knowledge about network topology other than server sizes)
            self.initial_distribute(gm, ini_path, seed=seed)

            # Then, we refine the placement using label propagation. This will
            # also ensure that the servers do not exceed their qubit capacity.
            # The new placement is updated in place in ``gain_manager``.
            self.initial_refinement(
                gm,
                num_rounds=num_rounds,
                stop_parameter=stop_parameter
            )
            # After refining, no server should be overpopulated
            assert gm.placement.is_valid(dist_circ, network)

            # Uncoarsen until no more CoarsenedPairs are left in the stack; after
            # each uncoarsening step, the hidden vertex and its neighbours have
            # their placement refined
            while gm.hypergraph.memento:
                self.uncoarsen_and_refine(gm)
            assert not gm.hypergraph.hidden_vertices

            assert gm.placement.is_valid(dist_circ, network)
            return gm.placement

    def coarsen_once(self, gm: GainManager):
        """Find the best coarsening pair and contract them. The heuristic
        used to choose the pair is the "heavy edge" heuristic from Sebastian's
        thesis on KaHyPar. We always contract a gate vertex (hide it) in a
        qubit vertex (acting as representative). Hence, no qubit vertices are
        ever hidden.
        """
        best_rep = None
        best_to_hide = None
        best_value: float = 0
        # We iterate over every adjacent pair of qubit and gate vertex
        for rep in gm.hypergraph.qubit_vertices:
            for to_hide in gm.hypergraph.current_neighbours(rep):
                # Skip those that are qubit vertices
                if to_hide in gm.hypergraph.qubit_vertices: continue
                # Compute heavy edge heuristic.
                #   This heuristic promotes coarsening pairs of vertices that
                #   are connected by many small hyperedges. The reasoning is
                #   the following: we want to contract as many hyperedges as
                #   possible since we will attempt to allocate both vertices
                #   to the same server and, hence, we want as many hyperedges
                #   as we can manage to be local; we penalise large hyperedges
                #   since for them to be local all vertices must be allocated
                #   to the same server, which is less likely for larger
                #   hyperedges. This heuristic originally comes from KaHyPar,
                #   in particular, Sebastian's thesis; where it is discussed
                #   to be a good heuristic for coarsening general hypergraphs.
                value: float = 0
                for hedge_id in gm.hypergraph.hyperedge_dict[to_hide]:
                    if hedge_id in gm.hypergraph.hyperedge_dict[rep]:
                        hyperedge = gm.hypergraph.hyperedge_hash[hedge_id]
                        value += float(hyperedge.weight) / len(hyperedge.vertices)
                # Update if better
                if value > best_value:
                    best_rep = rep
                    best_to_hide = to_hide
                    best_value = value

        assert best_rep is not None and best_to_hide is not None
        gm.hypergraph.coarsen(best_rep, best_to_hide)

    def uncoarsen_and_refine(self, gm: GainManager):
        """Uncoarsen the last pair and refine its placement and that of
        its neighbours.
        """
        rep, hidden = gm.hypergraph.uncoarsen()
        # The vertices whose placement is to be refined are the ``hidden`` one
        # and its neighbours (the latter in random order)
        neighbours = list(gm.hypergraph.current_neighbours(hidden))
        random.shuffle(neighbours)
        active_vertices = [hidden] + neighbours
        # Refine the placement of each of these vertices
        for vertex in active_vertices:
            self.apply_best_move(gm, vertex)

    def initial_refinement(self, gm: GainManager, **kwargs):
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

        :key num_rounds: Max number of refinement rounds. Default is 1000.
        :key stop_parameter: Real number in [0,1]. If proportion of moves
            in a round is smaller than this number, do no more rounds. Default
            is 0.05.
        """
        num_rounds = kwargs.get("num_rounds", 1000)
        stop_parameter = kwargs.get("stop_parameter", 0.05)

        # Since KaHyPar does not guarantee that the requirement on server
        # capacity will be satisfied, we enforce this ourselves.
        # However, it usually does satisfy the requirement and the following
        # code often does nothing or it moves very few vertices.
        for vertex in gm.hypergraph.qubit_vertices:
            # If moving in place is not valid then the current server is full
            if not gm.is_move_valid(
                vertex, gm.current_server(vertex)
            ):
                # Then, find a server where the move would be valid
                for server in gm.occupancy.keys():
                    if gm.is_move_valid(vertex, server):
                        # Move ``vertex`` to a server with free spaces and
                        # return control to the outer loop
                        gm.move(vertex, server)
                        break
        # Notice that the moves have been arbitrary, i.e. we have not
        # calculated gains. This is fine since the vertices we moved will
        # likely be boundary vertices; the following rounds of the
        # refinement algorithm will move them around to optimise gains.

        round_id = 0
        proportion_moved: float = 1
        while round_id < num_rounds and proportion_moved > stop_parameter:
            active_vertices = gm.hypergraph.get_boundary(gm.placement)

            moves = 0
            for vertex in active_vertices:
                # Attempt to apply the best move
                if self.apply_best_move(gm, vertex):
                    # The placement in ``gain_manager`` has been updated
                    moves += 1

            round_id += 1
            proportion_moved = (
                moves / len(active_vertices) if active_vertices else 0
            )

    def apply_best_move(self, gm: GainManager, vertex: int) -> bool:
        """TODO
        """
        current_server = gm.current_server(vertex)
        # We only consider moving ``vertex`` to a server that has
        # a neighbour vertex allocated to it
        potential_servers = set(
            gm.current_server(v)
            for v in gm.hypergraph.current_neighbours(vertex)
        )
        # We explicitly add the current server to the set
        # i.e. a potentially valid move is doing no move at all
        potential_servers.add(gm.current_server(vertex))

        best_server = None
        best_gain = float("-inf")
        best_best_swap = None
        for server in potential_servers:
            # Servers that are not in ``potential_servers`` will always
            # have the worst gain since they contain no neighbours
            # of ``vertex``. As such, we  simply ignore them.

            gain = gm.gain(vertex, server)

            # If the move is not valid (i.e. the server is full) we
            # find the best vertex in ``server`` to swap this one with
            best_swap_vertex = None
            if not gm.is_move_valid(vertex, server):
                # The only vertices we can swap with are qubit ones
                # so that the occupancy of the server is maintained
                vs = gm.placement.get_vertices_in(server)
                valid_swaps = [
                    vertex
                    for vertex in vs
                    if vertex in gm.hypergraph.qubit_vertices
                ]

                # To obtain the gain accurately, we move ``vertex`` to
                # ``server`` first and move it back at the end. This is
                # possible because ``move`` is an unsafe function, i.e.
                # it does not require that the move is valid.
                gm.move(vertex, server)

                best_swap_gain = float("-inf")
                for swap_vertex in valid_swaps:
                    swap_gain = gm.gain(
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
                gm.move(vertex, current_server)

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
            gm.move(vertex, best_server)
            if best_best_swap is not None:
                # This means that the move was not valid, so we need
                # to swap to make it valid
                gm.move(best_best_swap, current_server)
            # Report whether a move was applied or not
            return True
        else: return False

    def initial_distribute(self, gm: GainManager, ini_path: str, **kwargs):
        """Distribute ``hypergraph`` onto ``network`` using graph partitioning
        tools available in `kahypar <https://kahypar.org/>`_ package. The
        placement returned is not taking into account network topology.
        However, it does take into account server sizes.
        The chosen initial placement is set in ``gain_manager``.

        :key seed: Seed for the random number generator
        """
        seed = kwargs.get("seed", None)

        hyperedge_indices, hyperedges = gm.hypergraph.kahypar_hyperedges()

        num_hyperedges = len(hyperedge_indices) - 1
        num_vertices = len(list(set(hyperedges)))
        server_list = gm.network.get_server_list()
        num_servers = len(server_list)
        server_sizes = [len(gm.network.server_qubits[s]) for s in server_list]
        # For now, all hyperedges are assumed to have the same weight
        hyperedge_weights = [1 for i in range(0, num_hyperedges)]
        # After coarsening, all non-hidden vertices are qubit vertices:
        assert set(gm.hypergraph.vertex_list) - gm.hypergraph.hidden_vertices == gm.hypergraph.qubit_vertices
        assert num_vertices == len(gm.hypergraph.qubit_vertices)
        # All qubit vertices are given weight 1; there are no gate vertices
        vertex_weights = [1 for i in range(0, num_vertices)]

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

        placement_dict = {i: server for i,
                          server in enumerate(partition_list)}
        # The current placement is only for qubit vertices; place all gate
        # vertices in the same server as the vertex they coarsened with
        for coarsened_pair in gm.hypergraph.memento:
            placement_dict[coarsened_pair.hidden] = placement_dict[coarsened_pair.representative]
        # Set the initial placement
        gm.set_initial_placement(Placement(placement_dict))
