from __future__ import annotations

import networkx as nx  # type: ignore
from pytket_dqc.utils import steiner_tree
from pytket_dqc.circuits.hypergraph import Hyperedge

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pytket_dqc import Distribution


class GainManager:
    """Instances of this class are used to manage pre-computed values of the
    gain of a move, since it is likely that the same value will be used
    multiple times and computing it requires solving a minimum spanning tree
    problem which takes non-negligible computation time.

    :param distribution: The current state of the distribution
    :type distribution: Distribution
    :param qubit_vertices: The subset of vertices that correspond to qubits
    :type qubit_vertices: frozenset[int]
    :param server_graph: The nx.Graph of ``distribution.network``
    :type server_graph: nx.Graph
    :param occupancy: Maps servers to its current number of qubit vertices
    :type occupancy: dict[int, int]
    :param hyperedge_cost_map: Contains the current cost of each hyperedge
    :type hyperedge_cost_map: dict[Hyperedge, int]
    :param h_embedding_required: For each hyperedge, it indicates whether
        an H-embedding is required to implement it
    :type h_embedding_required: dict[Hyperedge, bool]
    :param steiner_cache: A dictionary of sets of servers to their
        steiner tree
    :type steiner_cache: dict[frozenset[int], nx.Graph]
    :param max_key_size: The maximum size of the set of servers whose cost is
        stored in cache. If there are N servers and m = ``max_key_size`` then
        the cache will store up to N^m values. If set to 0, cache is ignored.
        Default value is 5.
    :type max_key_size: int
    """

    def __init__(
        self, initial_distribution: Distribution, max_key_size: int = 5,
    ):
        self.distribution: Distribution = initial_distribution
        self.max_key_size: int = max_key_size

        dist_circ = initial_distribution.circuit
        self.qubit_vertices: frozenset[int] = frozenset(
            [v for v in dist_circ.vertex_list if dist_circ.is_qubit_vertex(v)]
        )
        self.server_graph: nx.Graph = self.distribution.network.get_server_nx()
        self.occupancy: dict[int, int] = dict()
        self.hyperedge_cost_map: dict[Hyperedge, int] = dict()
        self.h_embedding_required: dict[Hyperedge, bool] = dict()
        self.steiner_cache: dict[frozenset[int], nx.Graph] = dict()

        for server in self.distribution.network.server_qubits.keys():
            self.occupancy[server] = 0
        for vertex, server in self.distribution.placement.placement.items():
            if vertex in self.qubit_vertices:
                self.occupancy[server] += 1

        for hypedge in dist_circ.hyperedge_list:
            self.update_cost(hypedge)

    def update_cost(self, hyperedge: Hyperedge):
        """Updates ``hyperedge_cost_map`` and the caches ``steiner_cache`` and
        ``h_embedding_required``.
        """

        # Retrieve tree from cache or calculate it
        servers = frozenset(
            self.distribution.placement.placement[v]
            for v in hyperedge.vertices
        )
        if len(servers) <= self.max_key_size:
            if servers not in self.steiner_cache.keys():
                tree = steiner_tree(self.server_graph, list(servers))
                self.steiner_cache[servers] = tree
            else:
                tree = self.steiner_cache[servers]
        else:
            tree = steiner_tree(self.server_graph, list(servers))

        # Retrieve embedding information from cache or calculate it
        if hyperedge not in self.h_embedding_required.keys():
            self.h_embedding_required[
                hyperedge
            ] = self.distribution.circuit.h_embedding_required(hyperedge)

        # Update the cost of the hyperedge
        self.hyperedge_cost_map[hyperedge] = self.distribution.hyperedge_cost(
            hyperedge,
            server_tree=tree,
            h_embedding=self.h_embedding_required[hyperedge],
        )

    def split_hyperedge_gain(
        self,
        old_hyperedge: Hyperedge,
        new_hyperedge_list: list[Hyperedge]
    ) -> int:
        """Calculate the cost gain from splitting a hyperedge.
        This uses `hyperedge_cost_map`, a stored hyperedge cost
        dictionary to reduce cost recalculation. The cost may be
        negative, indicating an increase in the cost caused by splitting.

        :param old_hyperedge: Hyperedge to be split.
        :type old_hyperedge: Hyperedge
        :param new_hyperedge_list: List of hyperedges into which
        `old_hyperedge` should be split.
        :type new_hyperedge_list: list[Hyperedge]
        :return: Cost of splitting hyperedge as specified.
        :rtype: int
        """

        # TODO: Here and in other split and merge related functions,
        # I'm using that we can store the cost of hyperedges that
        # might not be in the hypergraph any more. Are we okay with that.
        # In the case of steiner trees we have capped the size of the
        # tree which can be stored. We might wish to do something similar here?
        current_cost = self.hyperedge_cost_map[old_hyperedge]
        new_cost = 0
        for hyperedge in new_hyperedge_list:
            if hyperedge in self.hyperedge_cost_map.keys():
                new_cost += self.hyperedge_cost_map[hyperedge]
            else:
                new_cost += self.distribution.hyperedge_cost(hyperedge)

        return current_cost - new_cost

    def split_hyperedge(
        self,
        old_hyperedge: Hyperedge,
        new_hyperedge_list: list[Hyperedge],
        recalculate_cost: bool = True
    ):
        """Split hyperedge `old_hyperedge` into hyperedges in
        `new_hyperedge_list`. This method utilises the
        `Hypergraph.split_hyperedge` method.

        :param old_hyperedge: Hyperedge to be split
        :type old_hyperedge: Hyperedge
        :param new_hyperedge_list: List of hyperedges into which
        `old_hyperedge` should be split.
        :type new_hyperedge_list: list[Hyperedge]
        :param recalculate_cost: Update dictionary of hyperedge costs,
        defaults to True
        :type recalculate_cost: bool, optional
        """

        self.distribution.circuit.split_hyperedge(
            old_hyperedge=old_hyperedge,
            new_hyperedge_list=new_hyperedge_list,
        )

        if recalculate_cost:
            for hypedge in new_hyperedge_list:
                self.update_cost(hypedge)

    def merge_hyperedge_gain(
        self,
        to_merge_hyperedge_list: list[Hyperedge]
    ) -> int:
        """Calculate the gain from merging a list of hyperedges.
        This uses `hyperedge_cost_map`, a stored hyperedge cost
        dictionary to reduce cost recalculation. The cost may be
        negative, indicating an increase in the cost caused by merging.

        :param to_merge_hyperedge_list: List of hyperedges to be merged.
        :type to_merge_hyperedge_list: list[Hyperedge]
        :return: Gain from merging hyperedges. This may be negative.
        :rtype: int
        """

        current_cost = sum(
            self.hyperedge_cost_map[hyperedge]
            for hyperedge in to_merge_hyperedge_list
        )

        # Create new hyperedge by merging given list.
        new_hyperedge = Hyperedge(
            vertices=list(
                set(
                    [
                        vertex
                        for hyperedge in to_merge_hyperedge_list
                        for vertex in hyperedge.vertices
                    ]
                )
            ),
            weight=to_merge_hyperedge_list[0].weight
        )

        # Take cost of hyperedge from hyperedge_cost_map if it exists there,

        # else calculate it.
        if new_hyperedge in self.hyperedge_cost_map.keys():
            new_cost = self.hyperedge_cost_map[new_hyperedge]
        else:
            new_cost = self.distribution.hyperedge_cost(new_hyperedge)

        return current_cost - new_cost

    def merge_hyperedge(
        self,
        to_merge_hyperedge_list: list[Hyperedge],
        recalculate_cost: bool = True
    ):
        """Merge `to_merge_hyperedge_list`, a list of given hyperedges
        and update `hyperedge_cost_map`, a stored hyperedge cost
        dictionary. This uses the `Hyperedge.merge_hyperedges` method.

        :param to_merge_hyperedge_list: List of hyperedges to merge.
        :type to_merge_hyperedge_list: list[Hyperedge]
        :param recalculate_cost: Determines if the hyperedge cost dictionary
        should be updated, defaults to True
        :type recalculate_cost: bool, optional
        """

        new_hyperedge = self.distribution.circuit.merge_hyperedge(
            to_merge_hyperedge_list=to_merge_hyperedge_list
        )

        if recalculate_cost:
            self.update_cost(new_hyperedge)

    def move_vertex_gain(self, vertex: int, new_server: int) -> int:
        """Compute the gain of moving ``vertex`` to ``new_server``. Instead
        of calculating the cost of the whole hypergraph using the new
        placement, we simply compare the previous cost of all hyperedges
        incident to ``vertex`` and substract their new cost.
        Positive gains mean improvement.

        :param vertex: The vertex that would be moved
        :type vertex: int
        :param new_server: The server ``vertex`` would be moved to
        :type: int

        :return: The improvement (may be negative) of the cost of the
            placement after applying the move.
        :rtype: int
        """

        # If the move is not changing servers, the gain is zero
        prev_server = self.distribution.placement.placement[vertex]
        if prev_server == new_server:
            return 0

        prev_cost_map = dict()
        prev_cost = 0
        for hypedge in self.distribution.circuit.hyperedge_dict[vertex]:
            prev_cost_map[hypedge] = self.hyperedge_cost_map[hypedge]
            prev_cost += self.hyperedge_cost_map[hypedge]

        self.move_vertex(vertex, new_server)  # This will recalculate the costs

        new_cost = 0
        for hypedge in self.distribution.circuit.hyperedge_dict[vertex]:
            new_cost += self.hyperedge_cost_map[hypedge]

        # Move back without recalculating costs
        self.move_vertex(vertex, prev_server, recalculate_cost=False)
        # Reassign previous costs
        for hypedge, cost in prev_cost_map.items():
            self.hyperedge_cost_map[hypedge] = cost

        return prev_cost - new_cost

    def move_vertex(
        self,
        vertex: int,
        server: int,
        recalculate_cost: bool = True
    ):
        """Moves ``vertex`` to ``server``, updating ``placement`` and
        ``occupancy`` accordingly.
        By default it updates the cost of the hyperedge, but this can
        be switched off via ``recalculate_cost``.
        Note: this operation is (purposefully) unsafe, i.e. it is not
        checked whether the move is valid or not. If unsure, you should
        call ``is_move_valid``.
        """

        # If a hyperedge requires embedding, moving a vertex that is contained
        # in the embedded hyperedge could cause issues: it may be that it is
        # no longer embeddable, so the hyperedge that required embedding can
        # no longer be implemented with the estimated cost.
        # To avoid this, we simply forbid placement moves once a hyperedge
        # embedded.
        if any(b for b in self.h_embedding_required.values()):
            raise Exception("Changing the placement after gates are embedded \
                             is not allowed.")

        placement_dict = self.distribution.placement.placement
        dist_circ = self.distribution.circuit

        # Ignore if the move would leave in the same server
        if placement_dict[vertex] != server:

            if vertex in self.qubit_vertices:
                self.occupancy[server] += 1
                self.occupancy[placement_dict[vertex]] -= 1

            placement_dict[vertex] = server

            if recalculate_cost:
                for hypedge in dist_circ.hyperedge_dict[vertex]:
                    self.update_cost(hypedge)

    def is_move_valid(self, vertex: int, server: int) -> bool:
        """ The move is only invalid when ``vertex`` is a qubit vertex and
        ``server`` is at its maximum occupancy. Notice that ``server`` may
        be where ``vertex`` was already placed.
        """
        if vertex in self.qubit_vertices:
            capacity = len(self.distribution.network.server_qubits[server])

            if server == self.current_server(vertex):
                return self.occupancy[server] <= capacity
            else:
                return self.occupancy[server] < capacity

        # Gate vertices can be moved freely
        else:
            return True

    def current_server(self, vertex: int):
        """Return the server that ``vertex`` is placed at.
        """
        return self.distribution.placement.placement[vertex]

    def set_max_key_size(self, max_key_size: int):
        """Set the ``max_key_size`` parameter.
        """
        self.max_key_size = max_key_size
