from pytket_dqc.circuits import HypergraphCircuit, Hyperedge
from pytket_dqc.placement import Placement
from pytket_dqc.networks import NISQNetwork
from pytket_dqc.utils import steiner_tree
from pytket import Circuit, OpType
import networkx as nx  # type: ignore
from numpy import isclose  # type: ignore


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
        return self.placement.is_valid(
            self.circuit, self.network
        )

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

    def to_pytket_circuit(self) -> Circuit:
        if self.is_valid():
            raise Exception("The distribution of the circuit is not valid!")
        raise NotImplementedError
