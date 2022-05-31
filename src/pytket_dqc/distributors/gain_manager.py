from __future__ import annotations

import networkx as nx  # type: ignore
from networkx.algorithms.approximation.steinertree import (  # type: ignore
    steiner_tree,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pytket_dqc.placement import Placement
    from pytket_dqc.networks import NISQNetwork
    from pytket_dqc.circuits import DistributedCircuit

class GainManager:
    """Instances of this class are used to manage pre-computed values of the
    gain of a move, since it is likely that the same value will be used
    multiple times and computing it requires solving a minimum spanning tree
    problem which takes non-negligible computation time.

    :param dist_circ: The circuit to be distributed, carries hypergraph info
    :type dist_circ: DistributedCircuit
    :param network: The network topology that the circuit must be mapped to
    :type network: NISQNetwork
    :param server_graph: The nx.Graph of ``network``
    :type server_graph: nx.Graph
    :param placement: The current placement
    :type placement: Placement
    :param occupancy: Maps servers to its current number of qubit vertices
    :type occupancy: dict[int, int]
    :param cache: A dictionary of sets of servers to their communication cost
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
        self.server_graph: nx.Graph = network.get_server_nx()
        self.placement: Placement = placement
        self.cache: dict[frozenset[int], int] = dict()
        self.occupancy: dict[int, int] = dict()

        for vertex, server in placement.placement.items():
            if dist_circ.is_qubit_vertex(vertex):
                if server not in self.occupancy.keys():
                    self.occupancy[server] = 0
                self.occupancy[server] += 1

    def gain(self, vertex: int, new_server: int) -> int:
        """Compute the gain of moving ``vertex`` to ``new_server``. Instead
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

        # If the move is not changing servers, the gain is zero
        current_server = self.placement.placement[vertex]
        if current_server == new_server:
            return 0

        gain = 0
        loss = 0
        for hyperedge in self.dist_circ.hyperedge_dict[vertex]:
            # List of servers connected by ``hyperedge - {vertex}``
            connected_servers = [
                self.placement.placement[v]
                for v in hyperedge.vertices
                if v != vertex
            ]

            # Number of vertices from ``hyperedge`` in ``current_server``
            current_server_pins = len(
                [
                    v
                    for v in hyperedge.vertices
                    if self.placement.placement[v] == current_server
                ]
            )
            # Number of vertices from ``hyperedge`` in ``new_server``
            new_server_pins = len(
                [
                    v
                    for v in hyperedge.vertices
                    if self.placement.placement[v] == new_server
                ]
            )

            # The cost of hyperedge will only be decreased by the move if
            # ``vertex`` is the last member of ``hyperedge`` in
            # ``current_server``
            if current_server_pins == 1:
                gain += self.steiner_cost(
                    frozenset(connected_servers + [current_server])
                )

            # The cost of hyperedge will only be increased by the move if
            # no vertices from ``hyperedge`` were in ``new_server`` prior
            # to the move
            if new_server_pins == 0:
                loss += self.steiner_cost(
                    frozenset(connected_servers + [new_server])
                )

        return gain - loss

    def steiner_cost(self, servers: frozenset[int]) -> int:
        """Finds a Steiner tree connecting all ``servers`` and returns number
        of edges. Makes use of the cache if the cost has already been computed
        and otherwise updates it.

        :param servers: The servers to be connected by the Steiner tree.
            The set is required to be a frozenset so that it is hashable.
        :type servers: frozenset[int]

        :return: The cost of connecting ``servers``
        :rtype: int
        """
        if servers not in self.cache.keys():
            tree = steiner_tree(self.server_graph, servers)
            self.cache[servers] = len(tree.edges)

        return self.cache[servers]

    def move(self, vertex: int, server: int):
        """Moves ``vertex`` to ``server``, updating ``placement`` and
        ``occupancy`` accordingly.
        """
        self.occupancy[server] += 1
        self.occupancy[self.placement.placement[vertex]] -= 1
        self.placement.placement[vertex] = server

    def is_move_valid(self, vertex: int, server: int) -> bool:
        """ The move is only invalid when ``vertex`` is a qubit vertex and
        ``server`` is at its maximum occupancy. Notice that ``server`` may
        be where ``vertex`` was already placed.
        """
        if self.dist_circ.is_qubit_vertex(vertex):
            capacity = len(self.network.server_qubits[server])

            if server == self.current_server(vertex):
                return self.occupancy[server] <= capacity
            else:
                return self.occupancy[server] < capacity

        # Gate vertices can be moved freely
        else:
            return True

    def current_server(self, vertex: int):
        """Just an alias to make code clearer.
        """
        return self.placement.placement[vertex]
