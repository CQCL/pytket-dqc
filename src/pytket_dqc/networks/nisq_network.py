from .server_network import ServerNetwork
from itertools import combinations
import networkx as nx  # type:ignore
from pytket.routing import Architecture, NoiseAwarePlacement  # type:ignore
from pytket.circuit import Node  # type:ignore
from typing import Tuple


class NISQNetwork(ServerNetwork):

    def __init__(
        self,
        server_coupling: list[list[int]],
        server_qubits: dict[int, list[int]]
    ):

        super().__init__(server_coupling)

        # Check that each server has a collection of qubits
        # which belong to it specified.
        for server in self.get_server_list():
            if server not in server_qubits.keys():
                raise Exception(
                    f"The qubits in server {server}"
                    " have not been specified."
                )

        # Combine all lists of qubits belonging to each server into one list.
        qubit_list = [
            qubit
            for qubit_list in server_qubits.values()
            for qubit in qubit_list
        ]
        # Check that each qubit belongs to only one server.
        if not len(qubit_list) == len(set(qubit_list)):
            raise Exception(
                "Qubits may belong to only one server"
                ", and should feature only once per server."
            )

        self.server_qubits = server_qubits

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
                G.add_edge(*qubit_connection, color="r", weight=0)

        for u, v in self.server_coupling:
            G.add_edge(
                self.server_qubits[u][0],
                self.server_qubits[v][0],
                color="b",
                weight=1,
            )

        return G

    def draw_nisq_network(self):

        G = self.get_nisq_nx()
        colors = [G[u][v]["color"] for u, v in G.edges()]
        nx.draw(G, with_labels=True, edge_color=colors)
