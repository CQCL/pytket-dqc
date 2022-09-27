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
    :param hyperedge_cost_map: Contains the current cost of each hyperedge
    :type hyperedge_cost_map: dict[Hyperedge, int]
    :param h_embedding_required: The subset of hyperedges that require
        H-embeddings to be implemented
    :type h_embedding_required: list[Hyperedge]
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
        self.hyperedge_cost_map = dict()
        self.h_embedding_required = []
        self.steiner_cache: dict[frozenset[int], nx.Graph] = dict()

        for server in self.distribution.network.server_qubits.keys():
            self.occupancy[server] = 0
        for vertex, server in self.distribution.placement.placement.items():
            if vertex in self.qubit_vertices:
                self.occupancy[server] += 1

        for hypedge in dist_circ.hyperedge_list:
            if dist_circ.h_embedding_required(hypedge):
                self.h_embedding_required.append(hypedge)

            self.hyperedge_cost_map[hypedge] = self.hyperedge_cost(hypedge)

    def gain(self, vertex: int, new_server: int) -> int:
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

        self.move(vertex, new_server)  # This will recalculate the costs

        new_cost = 0
        for hypedge in self.distribution.circuit.hyperedge_dict[vertex]:
            new_cost += self.hyperedge_cost_map[hypedge]

        # Move back without recalculating costs
        self.move(vertex, prev_server, recalculate_cost=False)
        # Reassign previous costs
        for hypedge, cost in prev_cost_map.items():
            self.hyperedge_cost_map[hypedge] = cost

        return prev_cost - new_cost

    def move(self, vertex: int, server: int, recalculate_cost=True):
        """Moves ``vertex`` to ``server``, updating ``placement`` and
        ``occupancy`` accordingly.
        By default it updates the cost of the hyperedge, but this can
        be switched off via ``recalculate_cost``.
        Note: this operation is (purposefully) unsafe, i.e. it is not
        checked whether the move is valid or not. If unsure, you should
        call ``is_move_valid``.
        """
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
                    self.hyperedge_cost_map[hypedge] = self.hyperedge_cost(
                        hypedge
                    )

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

    def hyperedge_cost(self, hyperedge: Hyperedge) -> int:
        """First, we check whether the hyperedge requires H-embeddings to be
        implemented. If not, we calculate its cost by counting the number of
        edges in the Steiner tree connecting all required servers. Otherwise,
        the cost of implementing the hyperedge is calculated using an "as lazy
        as possible" (ALAP) algorithm which we expect not to be optimal, but
        decent enough. In the latter acse, both reduction of ebit cost via
        Steiner trees and embedding are considered.

        Note: the cost only takes into account the ebits required to
        distribute the gates in the hyperedge; it does not consider the ebit
        cost of distributing the embedded gates. However, it does guarantee
        that the correction gates added during the embedding will not require
        extra ebits when distributed.
        """
        if hyperedge.weight != 1:
            raise Exception(
                "Hyperedges with weight other than 1 \
                 are not currently supported"
            )

        dist_circ = self.distribution.circuit
        placement_map = self.distribution.placement.placement

        # Extract hyperedge data
        shared_qubit = dist_circ.get_qubit_vertex(hyperedge)
        home_server = placement_map[shared_qubit]
        servers = frozenset(placement_map[v] for v in hyperedge.vertices)
        # Obtain the Steiner tree, retrieve it from the cache if possible
        if len(servers) <= self.max_key_size:
            if servers not in self.steiner_cache.keys():
                tree = steiner_tree(self.server_graph, servers)
                self.steiner_cache[servers] = tree
            else:
                tree = self.steiner_cache[servers]
        else:
            tree = steiner_tree(self.server_graph, servers)

        # If H-embedding is not required, we can easily calculate the cost
        if hyperedge not in self.h_embedding_required:
            return len(tree.edges)

        # Otherwise, we need to run ALAP
        else:
            # Collect all of the commands between the first and last gates in
            # the hyperedge. Ignore those do not act on the shared qubit.
            commands = dist_circ.get_hyperedge_subcircuit(hyperedge)

            # We will use the fact that, by construction, the index of the
            # vertices is ordered (qubits first, then gates left to right)
            vertices = sorted(hyperedge.vertices.copy())
            assert vertices.pop(0) == shared_qubit

            cost = 0
            currently_embedding = False  # Switched when finding a Hadamard
            connected_servers = {home_server}  # Servers shared_qubit access
            for command in commands:

                if command.op.type == OpType.H:
                    currently_embedding = not currently_embedding

                elif command.op.type in [OpType.X, OpType.Z]:
                    pass  # These gates can always be embedded

                elif command.op.type == OpType.Rz:
                    assert (
                        not currently_embedding
                        or isclose(command.op.params[0] % 1, 0)  # Identity
                        or isclose(command.op.params[0] % 1, 1)  # Z gate
                    )

                elif command.op.type == OpType.CU1:

                    if currently_embedding:  # Gate to be embedded
                        assert isclose(command.op.params[0] % 2, 1)  # CZ gate

                        qubits = [
                            dist_circ.qubit_to_vertex_map[q]
                            for q in command.qubits
                        ]
                        remote_qubit = [
                            q for q in qubits if q != shared_qubit
                        ][0]
                        remote_server = placement_map[remote_qubit]

                        # According to the condition for embeddability on
                        # multiple servers, we required that ``remote_server``
                        # has access to an ebit sharing ``shared_qubit``
                        assert remote_server in connected_servers

                        # Only servers in the connection path are left intact
                        # all others need to be disconnected since, otherwise,
                        # extra ebits would be required to implement the new
                        # correction gates that would be introduced
                        connection_path = nx.shortest_path(
                            tree, home_server, remote_server
                        )
                        connected_servers = connected_servers.intersection(
                            connection_path
                        )
                        # Note: we do not need to consider the ebits required
                        # to implement the embedded gate since that one is not
                        # within this hyperedge. Thus, we are done here.

                    else:  # Gate to be distributed (or already local)

                        # Get the server where the gate is to be implemented
                        gate_vertex = vertices.pop(0)
                        assert (
                            dist_circ.vertex_circuit_map[gate_vertex][
                                "command"
                            ]
                            == command
                        )
                        gate_server = placement_map[gate_vertex]
                        # If gate_server doesn't have access to shared_qubit
                        # update the cost, adding the necessary ebits
                        if gate_server not in connected_servers:
                            connection_path = set(
                                nx.shortest_path(
                                    tree, home_server, gate_server
                                )
                            )
                            required_connections = connection_path.difference(
                                connected_servers
                            )
                            connected_servers.update(required_connections)
                            cost += len(required_connections)
            return cost

    def current_server(self, vertex: int):
        """Return the server that ``vertex`` is placed at.
        """
        return self.distribution.placement.placement[vertex]

    def set_max_key_size(self, max_key_size: int):
        """Set the ``max_key_size`` parameter.
        """
        self.max_key_size = max_key_size
