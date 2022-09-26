from __future__ import annotations

import networkx as nx  # type: ignore
from numpy import isclose  # type: ignore
from networkx.algorithms.approximation.steinertree import (  # type: ignore
    steiner_tree,
)
from pytket import OpType

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pytket_dqc import Distribution
    from pytket_dqc.circuits import Hyperedge


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
    :param steiner_cache: A dictionary of sets of servers to their
        communication cost
    :type steiner_cache: dict[frozenset[int], int]
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
        self.steiner_cache: dict[frozenset[int], int] = dict()

        for server in self.distribution.network.server_qubits.keys():
            self.occupancy[server] = 0
        for vertex, server in self.distribution.placement.placement.items():
            if vertex in self.qubit_vertices:
                self.occupancy[server] += 1

    def gain(self, vertex: int, new_server: int) -> int:
        """Compute the gain of moving ``vertex`` to ``new_server``. Instead
        of calculating the cost of the whole hypergraph using the new
        placement, we simply compare the previous cost of all hyperedges
        incident to ``vertex`` and substract their new cost. Moreover, if
        these values are available in the cache they are used; otherwise,
        the cache is updated.
        The costs of each hyperedge are calculated using Steiner trees over
         ``network``.
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
        current_server = self.distribution.placement.placement[vertex]
        if current_server == new_server:
            return 0

        gain = 0
        for hyperedge in self.distribution.circuit.hyperedge_dict[vertex]:
            # List of servers connected by ``hyperedge - {vertex}``
            connected_servers = [
                self.distribution.placement.placement[v]
                for v in hyperedge.vertices
                if v != vertex
            ]

            current_cost = self.steiner_cost(
                frozenset(connected_servers + [current_server])
            )
            new_cost = self.steiner_cost(
                frozenset(connected_servers + [new_server])
            )
            gain += hyperedge.weight * (current_cost - new_cost)

        return gain

    def hyperedge_cost(self, hyperedge: Hyperedge) -> int:
        """The cost implementing the hyperedge is calculated using an "as lazy
        as possible" (ALAP) algorithm which we expect not to be optimal, but
        decent enough. Both reduction of ebit cost via Steiner trees and
        embedding are considered.

        Note: the cost only takes into account the ebits required to
        distribute the gates in the hyperedge; it does not consider the ebit
        cost of distributing the embedded gates. However, it does guarantee
        that the correction gates added during the embedding will not require
        extra ebits when distributed.
        """
        if hyperedge.weight != 1:
            raise Exception("Hyperedges with weight other than 1 are not currently supported")

        dist_circ = self.distribution.circuit
        placement_map = self.distribution.placement.placement

        # Extract hyperedge data
        shared_qubit = dist_circ.get_qubit_vertex(hyperedge)
        home_server = placement_map[shared_qubit]
        connected_servers = [
            placement_map[v]
            for v in hyperedge.vertices
        ]
        # Obtain the Steiner tree, retrieve it from the cache if possible
        if len(servers) <= self.max_key_size:
            if servers not in self.steiner_cache.keys():
                steiner_tree = steiner_tree(self.server_graph, servers)
                self.steiner_cache[servers] = steiner_tree
            else:
                steiner_tree = self.steiner_cache[servers]
        else:
            steiner_tree = steiner_tree(self.server_graph, servers)

        # Collect all of the commands between the first and last gates in the
        # hyperedge. Ignore the commands that do not act on the shared qubit.
        commands = dist_circ.get_hyperedge_subcircuit(hyperedge)

        cost = 0
        currently_embedding = False  # Switched when encountering a Hadamard
        connected_servers = {home_server}  # Servers with shared_qubit access
        for command in commands:

            if command.op.type == OpType.H:
                currently_embedding = not currently_embedding

            elif command.op.type in [OpType.X, OpType.Z]:
                pass  # These gates can always be embedded

            elif command.op.type == OpType.Rz:
                assert isclose(command.op.params[0] % 2, 1) or not currently_embedding

            elif command.op.type == OpType.CU1:
                other_qubits = [q for q in command.qubits if q != shared_qubit]
                assert len(other_qubits) == 1
                remote_qubit = dist_circ.qubit_to_vertex_map[other_qubits[0]]
                remote_server = placement_map[remote_qubit]

                if currently_embedding:  # Gate to be embedded
                    assert isclose(command.op.params[0] % 2, 1)  # CZ gate

                    # Only servers in the connection path are left connected
                    # all others need to be disconnected since, otherwise,
                    # extra ebits would be required to implement the new
                    # correction gates that would be introduced
                    connection_path = nx.shortest_path(steiner_tree, home_server, remote_server)
                    connected_servers = connected_servers.intersection(connection_path)
                    # Note: we do not need to consider the ebits required
                    # to implement the embedded gate since that one is not
                    # within this hyperedge. Thus, we are done here.

                else:  # Gate to be distributed (or already local)
                    # If the remote_server doesn't have access to shared_qubit
                    # update the cost, adding the necessary ebits
                    if remote_server not in connected_servers:
                        connection_path = set(nx.shortest_path(steiner_tree, home_server, remote_server))
                        required_connections = connection_path.difference(connected_servers)
                        connected_servers.updated(required_connections)
                        cost += len(required_connections)
        return cost



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
        if len(servers) <= self.max_key_size:
            if servers not in self.steiner_cache.keys():
                tree = steiner_tree(self.server_graph, servers)
                self.steiner_cache[servers] = len(tree.edges)
            cost = self.steiner_cache[servers]
        else:
            tree = steiner_tree(self.server_graph, servers)
            cost = len(tree.edges)

        return cost

    def move(self, vertex: int, server: int):
        """Moves ``vertex`` to ``server``, updating ``placement`` and
        ``occupancy`` accordingly. Note: this operation is (purposefully)
        unsafe, i.e. it is not checked whether the move is valid or not.
        If unsure, you should call ``is_move_valid``.
        """
        if vertex in self.qubit_vertices:
            self.occupancy[server] += 1
            self.occupancy[self.distribution.placement.placement[vertex]] -= 1

        self.distribution.placement.placement[vertex] = server

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
