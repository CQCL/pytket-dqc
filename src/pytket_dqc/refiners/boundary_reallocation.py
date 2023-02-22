from __future__ import annotations

import random
from pytket_dqc.allocators import GainManager
from pytket_dqc.refiners import Refiner

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pytket_dqc import Distribution


class BoundaryReallocation(Refiner):
    """Refiner that considers reallocations of the vertices on the boundary of
    the partition, attempting to improve the solution. It is greedy and it
    is not capable of escaping local optima. The justification
    is that we assume that the Allocator used already grouped qubits
    efficiently and that our network has short average distance, so that we
    only need to refine the allocation of vertices in the boundary.
    """

    def refine(self, distribution: Distribution, **kwargs) -> bool:
        """The refinement algorithm proceeds in rounds. In each round, all of
        the vertices in the boundary are visited in random order and we
        we calculate the gain achieved by moving the vertex to other servers.
        The best move is applied, with ties broken randomly. If all possible
        moves have negative gains, the vertex is not moved.

        The algorithm continues until the proportion of vertices
        moved in a round (i.e. #moved / #boundary) is smaller than
        ``stop_parameter`` or the maximum ``num_rounds`` is reached.

        This refinement algorithm is taken from the discussion of the "label
        propagation" algorithm in https://arxiv.org/abs/1402.3281. However,
        we do not consider any coarsening.

        :param distribution: Distribution to refine.
        :type distribution: Distribution

        :key reallocate_qubits: Whether qubit vertices are allowed to be
            reallocated. Default is True.
        :key num_rounds: Max number of refinement rounds. Default is 1000.
        :key stop_parameter: Real number in [0,1]. If proportion of moves
            in a round is smaller than this number, do no more rounds. Default
            is 0.05.
        :key seed: Seed for randomness. Default is None.
        :key cache_limit: The maximum size of the set of servers whose cost is
            stored in cache; see GainManager. Default value is 5.

        :return: Distribution where the placement updated.
        :rtype: Distribution
        """

        reallocate_qubits = kwargs.get("reallocate_qubits", True)
        num_rounds = kwargs.get("num_rounds", 10)
        stop_parameter = kwargs.get("stop_parameter", 0.05)
        seed = kwargs.get("seed", None)
        if seed is not None:
            random.seed(seed)
        cache_limit = kwargs.get("cache_limit", None)

        # We will use a ``GainManager`` to manage the calculation of gains
        # (and management of pre-computed values) in a transparent way
        gain_manager = GainManager(distribution)
        if cache_limit is not None:
            gain_manager.set_max_key_size(cache_limit)

        round_id = 0
        proportion_moved: float = 1
        refinement_made = False
        dist_circ = gain_manager.distribution.circuit
        placement = gain_manager.distribution.placement
        while round_id < num_rounds and proportion_moved > stop_parameter:
            active_vertices = dist_circ.get_boundary(placement)
            if not reallocate_qubits:
                active_vertices = [
                    v
                    for v in active_vertices
                    if not dist_circ.is_qubit_vertex(v)
                ]

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

                    gain = gain_manager.move_vertex_gain(vertex, server)

                    # If the move is not valid (i.e. the server is full) we
                    # find the best vertex in ``server`` to swap this one with
                    best_swap_vertex = None
                    if not gain_manager.is_move_valid(vertex, server):
                        # The only vertices we can swap with are qubit ones
                        # so that the occupancy of the server is maintained

                        vs = placement.get_vertices_in(server)
                        valid_swaps = [
                            vertex
                            for vertex in vs
                            if dist_circ.is_qubit_vertex(vertex)
                        ]

                        # To obtain the gain accurately, we move ``vertex`` to
                        # ``server`` first and move it back at the end. This is
                        # possible because ``move`` is an unsafe function, i.e.
                        # it does not require that the move is valid.
                        gain_manager.move_vertex(
                            vertex, server, recalculate_cost=False
                        )

                        best_swap_gain = float("-inf")
                        for swap_vertex in valid_swaps:
                            swap_gain = gain_manager.move_vertex_gain(
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
                        gain_manager.move_vertex(
                            vertex, current_server, recalculate_cost=False
                        )

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
                    gain_manager.move_vertex(vertex, best_server)
                    if best_best_swap is not None:
                        # This means that the move was not valid, so we need
                        # to swap to make it valid
                        gain_manager.move_vertex(
                            best_best_swap, current_server
                        )
                    refinement_made = True
                    # Either if we swap or we don't, we count it as one move
                    # since this is meant to count 'rounds with change' rather
                    # than literal moves
                    moves += 1

            round_id += 1
            proportion_moved = (
                moves / len(active_vertices) if active_vertices else 0
            )

        assert gain_manager.distribution.is_valid()
        # GainManager has updated ``distribution`` in place:
        assert gain_manager.distribution is distribution

        return refinement_made
