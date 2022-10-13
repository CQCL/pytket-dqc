from pytket_dqc.circuits import HypergraphCircuit, Hyperedge
from pytket_dqc.placement import Placement
from pytket_dqc.networks import NISQNetwork
from pytket_dqc.utils import steiner_tree, check_equivalence
from pytket_dqc.utils.gateset import start_proc, end_proc
from pytket_dqc.utils.circuit_analysis import _cost_from_circuit
from pytket import Circuit, OpType, Qubit
import networkx as nx  # type: ignore
from numpy import isclose  # type: ignore
from typing import NamedTuple


class Distribution:
    """Class containing all information required to generate pytket circuit.
    """

    def __init__(
        self,
        circuit: HypergraphCircuit,
        placement: Placement,
        network: NISQNetwork,
    ):
        """Initialisation function for Distribution

        :param circuit: Circuit to be distributed, including its hypergraph.
        :type circuit: HypergraphCircuit
        :param placement: A placement of the qubits and gates onto servers.
        :type placement: Placement
        :param network: Network onto which circuit is distributed.
        :type network: NISQNetwork
        :raises Exception: Raised if the placement is not valid for the
            circuit.
        :raises Exception: Raised if the placement is not valid for the
            packets.
        """

        self.circuit = circuit
        self.placement = placement
        self.network = network

    def is_valid(self) -> bool:
        """Check that this distribution can be implemented.
        """

        # TODO: There may be some other checks that we want to do here to check
        # that the packets hypergraph is not totally nonsensical. For example
        # that gate nodes are not in too many packets. I think the conditions
        # that are most relevant will emerge as the to_pytket_circuit method
        # is written.
        return self.placement.is_valid(self.circuit, self.network)

    def cost(self) -> int:
        """Return the number of ebits required for this distribution.
        """
        if not self.is_valid():
            raise Exception("This is not a valid distribution")

        cost = 0
        for hyperedge in self.circuit.hyperedge_list:
            cost += self.hyperedge_cost(hyperedge)
        return cost

    def hyperedge_cost(self, hyperedge: Hyperedge, **kwargs) -> int:
        """First, we check whether the hyperedge requires H-embeddings to be
        implemented. If not, we calculate its cost by counting the number of
        edges in the Steiner tree connecting all required servers. Otherwise,
        the cost of implementing the hyperedge is calculated using an "as lazy
        as possible" (ALAP) algorithm which we expect not to be optimal, but
        decent enough. In the latter case, both reduction of ebit cost via
        Steiner trees and embedding are considered.

        Note: the cost only takes into account the ebits required to
        distribute the gates in the hyperedge; it does not consider the ebit
        cost of distributing the embedded gates, since that will be calculated
        by calling ``hyperedge_cost`` that the embedded gates belong to.
        However, we do guarantee that the correction gates added during the
        embedding will not require extra ebits when distributed.

        :param hyperedge: The hyperedge whose cost is to be calculated.
        :type hyperedge: Hyperedge

        :key server_tree: The connectivity tree (subgraph of ``server_graph``)
            that should be used. If not provided, this function will find a
            Steiner tree.
        :type server_tree: nx.Graph
        :key server_graph: The network's ``server_graph``. If not provided,
            this is calculated from ``self.network``. Meant to save a call.
        :type server_graph: nx.Graph
        :key h_embedding: Indicates whether or not the hyperedge requires
            an H-embedding to be implemented. If not provided, it is checked.
        :type h_embedding: bool

        :return: The cost of the hyperedge.
        :rtype: int
        """
        tree = kwargs.get("server_tree", None)
        server_graph = kwargs.get("server_graph", None)
        h_embedding = kwargs.get("h_embedding", None)

        if hyperedge.weight != 1:
            raise Exception(
                "Hyperedges with weight other than 1 \
                 are not currently supported"
            )

        dist_circ = self.circuit
        placement_map = self.placement.placement

        # Extract hyperedge data
        shared_qubit = dist_circ.get_qubit_vertex(hyperedge)
        home_server = placement_map[shared_qubit]
        servers = [placement_map[v] for v in hyperedge.vertices]
        # Obtain the server graph if it is not given
        if server_graph is None:
            server_graph = self.network.get_server_nx()
        # Obtain the Steiner tree or check that the one given is valid
        if tree is None:
            tree = steiner_tree(server_graph, servers)
        else:
            assert all(s in tree.nodes for s in servers)

        # If not known, check if H-embedding is required for this hyperedge
        if h_embedding is None:
            h_embedding = dist_circ.h_embedding_required(hyperedge)

        # If H-embedding is not required, we can easily calculate the cost
        if not h_embedding:
            return len(tree.edges)

        # Otherwise, we need to run ALAP
        else:
            # Collect all of the commands between the first and last gates in
            # the hyperedge. Ignore those do not act on the shared qubit.
            commands = dist_circ.get_hyperedge_subcircuit(hyperedge)

            # We will use the fact that, by construction, the index of the
            # vertices is ordered (qubits first, then gates left to right)
            vertices = sorted(hyperedge.vertices)
            assert vertices.pop(0) == shared_qubit

            cost = 0
            currently_embedding = False  # Switched when finding a Hadamard
            connected_servers = {home_server}  # With access to shared_qubit
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
                            dist_circ.get_vertex_of_qubit(q)
                            for q in command.qubits
                        ]
                        remote_qubit = [
                            q for q in qubits if q != shared_qubit
                        ][0]
                        remote_server = placement_map[remote_qubit]

                        # Only servers in the shortest path from remote_server
                        # to home_server are left intact. The rest of the
                        # servers need to be disconnected since, otherwise,
                        # extra ebits would be required to implement the new
                        # correction gates that would be introduced.
                        #
                        # Note: the shortest path is found in server_graph
                        # instead of in the tree since that is how the
                        # embedded hyperedge would be implemented if it were
                        # not embedded (it only connects two servers, so its
                        # Steiner tree is the shortest path). If we used
                        # some other path then we would be changing the
                        # way the embedded gates are distributed, possibly
                        # increasing the cost of their distribution, hence,
                        # not following Junyi's main design principle.
                        #
                        # Note: by the conditions of embeddability, all gates
                        # that are being embedded simultaneously act on the
                        # same two servers.
                        connection_path = nx.shortest_path(
                            server_graph, home_server, remote_server
                        )
                        connected_servers = connected_servers.intersection(
                            connection_path
                        )

                    else:  # Gate to be distributed (or already local)

                        # Get the server where the gate is to be implemented
                        gate_vertex = vertices.pop(0)
                        assert (
                            dist_circ._vertex_circuit_map[gate_vertex][
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

    def get_qubit_mapping(self) -> dict[Qubit, Qubit]:
        """Mapping from circuit (logical) qubits to server (hardware) qubits.
        """
        if self.is_valid():
            raise Exception("The distribution of the circuit is not valid!")

        qubit_map = {}
        current_in_server = {s: 0 for s in self.network.get_server_list()}
        for v in self.circuit.get_qubit_vertices():
            server = self.placement.placement[v]
            circ_qubit = self.circuit.get_qubit_of_vertex(v)
            hw_qubit = Qubit(f"Server {server}", current_in_server[server])
            current_in_server[server] += 1
            qubit_map[circ_qubit] = hw_qubit

        return qubit_map

    def to_pytket_circuit(self) -> Circuit:
        if self.is_valid():
            raise Exception("The distribution of the circuit is not valid!")

        # -- SCOPE VARIABLES -- #
        # Accessible to the internal class below
        dist_circ = self.circuit
        server_graph = self.network.get_server_nx()
        placement_map = self.placement.placement
        qubit_mapping = self.get_qubit_mapping()

        # -- INTERNAL CLASSES -- #

        class EjppAction(NamedTuple):
            """Encodes the information to create Starting and EndingProcesses
            """

            starting: bool
            from_qubit: Qubit
            to_qubit: Qubit

        class LinkManager:
            """An internal class dedicated to managing the hardware qubits
            that store the ebits (i.e. link qubits).
            """

            def __init__(self, servers: list[int]):
                # A dictionary of serverId to list of available link qubits.
                self.available: dict[int, list[Qubit]] = {
                    s: [] for s in servers
                }
                # A dictionary of serverId to list of occupied link qubits.
                self.occupied: dict[int, list[Qubit]] = {
                    s: [] for s in servers
                }
                # A dictionary matching a currently hyperedge with a link to
                # a specified server with the link qubit in the server.
                self.link_qubit_dict: dict[tuple[Hyperedge, int], Qubit] = {}

            def next_available(self, server: int) -> Qubit:
                """Returns an available link qubit in ``server``. If there are
                none, it creates a new one.
                """
                if not self.available[server]:
                    next_id = len(self.occupied[server])
                    qubit = Qubit(f"Server {server} Link Register", next_id)
                else:
                    qubit = self.available[server].pop()

                return qubit

            def release(self, link_qubit: Qubit):
                """Releases ``link_qubit``, making it again available.
                """

                # Singleton list of servers containing ``link_qubit``
                servers = [
                    s for s, qs in self.occupied.items() if link_qubit in qs
                ]
                assert len(servers) == 1
                its_server = servers[0]

                self.occupied[its_server].remove(link_qubit)
                self.available[its_server].append(link_qubit)

                # Remove the entry from ``self.link_qubit_dict``
                keys = [
                    key
                    for key, q in self.link_qubit_dict.items()
                    if q == link_qubit
                ]
                assert len(keys) == 1
                its_key = keys[0]

                del self.link_qubit_dict[its_key]

            def start_link(
                self, target: int, hyperedge: Hyperedge
            ) -> list[EjppAction]:
                """Find the sequence of StartingProcesses required to share
                the qubit of ``hyperedge`` with the ``target`` server.
                """

                # Extract hyperedge data
                hyp_qubit = dist_circ.get_qubit_vertex(hyperedge)
                home_server = placement_map[hyp_qubit]
                hyp_servers = [placement_map[v] for v in hyperedge.vertices]
                tree = steiner_tree(server_graph, hyp_servers)
                assert target in hyp_servers

                # Find the path from the home_server to the target
                # NOTE: there is only one path because we search in a tree
                connection_path = nx.shortest_path(tree, home_server, target)
                # Remove the ``home_server`` from the connection path
                hs = connection_path.pop(0)
                assert hs == home_server

                # Go over the connection_path step by step adding the required
                # starting EjppActions
                starting_actions = []
                # We will keep track of the qubit that holds the copy of
                # ``hyp_qubit`` throughout the path of servers.
                # At the beginning, this is held by the HW qubit corresponding
                # to the ``hyp_qubit`` vertex
                last_link_qubit = qubit_mapping[
                    dist_circ.get_qubit_of_vertex(hyp_qubit)
                ]
                # For sanity check: after first disconnected server in the
                # path, all servers that follow should also be disconnected
                so_far_connected = True
                for next_server in connection_path:
                    if (hyperedge, next_server) in self.link_qubit_dict:
                        assert so_far_connected
                        last_link_qubit = self.link_qubit_dict[
                            (hyperedge, next_server)
                        ]
                    else:
                        so_far_connected = False
                        # Retrieve an available link qubit to populate
                        this_link_qubit = self.next_available(next_server)
                        # Add the EjppAction to create the connection
                        starting_actions.append(
                            EjppAction(
                                starting=True,
                                from_qubit=last_link_qubit,
                                to_qubit=this_link_qubit,
                            )
                        )
                        last_link_qubit = this_link_qubit

                return starting_actions

            def end_link(
                self, target: int, hyperedge: Hyperedge
            ) -> list[EjppAction]:
                """Find the sequence of EndingProcesses required to end the
                connection to the ``target`` server and all of its children.
                """

                # Extract hyperedge data
                hyp_qubit = dist_circ.get_qubit_vertex(hyperedge)
                home_server = placement_map[hyp_qubit]
                hyp_servers = [placement_map[v] for v in hyperedge.vertices]
                tree = steiner_tree(server_graph, hyp_servers)
                assert target in hyp_servers
                assert target != home_server

                # Find the children of ``target``, these also need to be
                # disconnected
                children_edges = nx.dfs_edges(tree, source=target)
                # The above produces a list of edges (parent, child).
                # Since the ``children_edges`` were generated via a DFS strat
                # (depth-first-search), if we reverse it we'll iterate from
                # the leave edges to the root edges.
                # Servers at the leaves of the tree must be disconnected first
                ending_actions = []
                for parent, child in reversed(children_edges):
                    # Sanity check
                    if (hyperedge, parent) not in self.link_qubit_dict:
                        assert (hyperedge, child) not in self.link_qubit_dict
                    # If the child is currently connected, disconnect it
                    elif (hyperedge, child) in self.link_qubit_dict:
                        parent_link = self.link_qubit_dict[(hyperedge, parent)]
                        child_link = self.link_qubit_dict[(hyperedge, child)]
                        # End the connection
                        ending_actions.append(
                            EjppAction(
                                starting=False,
                                from_qubit=child_link,
                                to_qubit=parent_link,
                            )
                        )
                        # Release the link qubit
                        self.release(child_link)

                # Don't forget that the ``target`` server itself must be
                # disconnected as well!
                directed_edges = nx.dfs_edges(tree, source=home_server)
                parents = [
                    parent
                    for parent, child in directed_edges
                    if child == target
                ]
                assert len(parents) == 1
                parent = parents[0]
                # If the parent is the ``home_server``, then the ``to_qubit``
                # of the EjppAction must be the original shared qubit
                if parent == home_server:
                    parent_link = qubit_mapping[
                        dist_circ.get_qubit_of_vertex(hyp_qubit)
                    ]
                # Otherwise, we find the link qubit of the parent
                else:
                    parent_link = self.link_qubit_dict[(hyperedge, parent)]
                    target_link = self.link_qubit_dict[(hyperedge, target)]
                # Disconnect the link qubit of ``target``
                ending_actions.append(
                    EjppAction(
                        starting=False,
                        from_qubit=target_link,
                        to_qubit=parent_link,
                    )
                )
                self.release(target_link)

                return ending_actions

        # -- CIRCUIT GENERATION -- #
        circ = Circuit()
        # Add the qubits to the circuit
        for hw_qubit in qubit_mapping.values():
            circ.add_qubit(hw_qubit)

        # Read the original circuit from left to right and, as we go:
        #   (1) add the required EJPP processes
        #   (2) distribute nonlocal gates
        #   (3) add the required correction gates for embedding
        raise NotImplementedError

        assert _cost_from_circuit(circ) == self.cost()
        assert check_equivalence(
            self.circuit.get_circuit(), circ, qubit_mapping
        )
        return circ
