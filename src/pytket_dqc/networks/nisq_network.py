from .server_network import ServerNetwork
from itertools import combinations
import networkx as nx  # type:ignore
from pytket.routing import Architecture, NoiseAwarePlacement  # type:ignore
from pytket.circuit import Node  # type:ignore
from typing import Tuple
import random


class NISQNetwork(ServerNetwork):

    def __init__(
        self,
        server_coupling: list[list[int]],
        server_qubits: dict[int, list[int]]
    ):

        super().__init__(server_coupling)

        self.server_qubits = server_qubits

        # Check that each server has a collection of qubits
        # which belong to it specified.
        for server in self.get_server_list():
            if server not in self.server_qubits.keys():
                raise Exception(
                    f"The qubits in server {server}"
                    " have not been specified."
                )

        qubit_list = self.get_qubit_list()
        # Check that each qubit belongs to only one server.
        if not len(qubit_list) == len(set(qubit_list)):
            raise Exception(
                "Qubits may belong to only one server"
                ", and should feature only once per server."
            )

        # Check that the resulting network is connected.
        if not nx.is_connected(self.get_nisq_nx()):
            raise Exception("This server network is unconnected.")

    def get_qubit_list(self) -> list[int]:

        # Combine all lists of qubits belonging to each server into one list.
        qubit_list = [
            qubit
            for qubit_list in self.server_qubits.values()
            for qubit in qubit_list
        ]

        return qubit_list

    def get_architecture(self) -> Tuple[Architecture, dict[Node, int]]:

        G = self.get_nisq_nx()
        arc = Architecture([(Node(u), Node(v)) for u, v in G.edges])
        node_qubit_map = {Node(u): u for u in G.nodes}

        return arc, node_qubit_map

    def get_placer(self) -> Tuple[
        Architecture,
        dict[Node, int],
        NoiseAwarePlacement
    ]:

        G = self.get_nisq_nx()
        link_errors = {}
        for u, v in G.edges:
            edge_data = G.get_edge_data(u, v)
            link_errors[(Node(u), Node(v))] = edge_data['weight']

        arc, node_qubit_map = self.get_architecture()

        return (
            arc,
            node_qubit_map,
            NoiseAwarePlacement(arc=arc, link_errors=link_errors)
        )

    def get_nisq_nx(self) -> nx.Graph:

        G = nx.Graph()

        for qubits in self.server_qubits.values():
            for qubit_connection in combinations(qubits, 2):
                G.add_edge(*qubit_connection, color="red", weight=0)

        for u, v in self.server_coupling:
            G.add_edge(
                self.server_qubits[u][0],
                self.server_qubits[v][0],
                color="blue",
                weight=1,
            )

        return G

    def draw_nisq_network(self):

        G = self.get_nisq_nx()
        colors = [G[u][v]["color"] for u, v in G.edges()]
        nx.draw(
            G,
            with_labels=True,
            edge_color=colors,
            pos=nx.nx_agraph.graphviz_layout(G)
        )


def random_connected_graph(n_nodes, edge_prob):

    if not (edge_prob >= 0 and edge_prob <= 1):
        raise Exception("edge_prob must be between 0 and 1.")

    edge_list = []

    # Build connected graph. Add each node to the graph
    # in sequence so that all are included in the graph.
    for node in range(1, n_nodes):
        edge_list.append({node, random.randrange(node)})

    # Randomly add additional edges with given probability.
    for node in range(n_nodes):
        for connected_node in [i for i in range(n_nodes) if i != node]:
            if random.random() < edge_prob:
                if {node, connected_node} not in edge_list:
                    edge_list.append({node, connected_node})

    G = nx.Graph()
    G.add_edges_from(edge_list)
    assert nx.is_connected(G)

    return [list(edge) for edge in edge_list]


class RandomNISQNetwork(NISQNetwork):

    def __init__(self, n_servers: int, n_qubits: int, edge_prob: float = 0.5):

        if n_qubits < n_servers:
            raise Exception(
                "The number of qubits must be greater ",
                "than the number of servers.")

        server_coupling = random_connected_graph(n_servers, edge_prob)

        server_qubits = {i: [i] for i in range(n_servers)}
        for qubit in range(n_servers, n_qubits):
            server = random.randrange(n_servers)
            server_qubits[server] = server_qubits[server] + [qubit]

        super().__init__(server_coupling, server_qubits)
