from .server_network import ServerNetwork
from itertools import combinations
import networkx as nx  # type:ignore


class NISQNetwork(ServerNetwork):
    def __init__(
        self,
        server_coupling: list[list[int]],
        server_qubits: dict[int, list[int]]
    ):

        super().__init__(server_coupling)

        for server_list in server_coupling:

            if server_list[0] not in server_qubits.keys():
                raise Exception(
                    f"The qubits in server {server_list[0]}"
                    " have not been specified."
                )

            if server_list[1] not in server_qubits.keys():
                raise Exception(
                    f"The qubits in server {server_list[1]}"
                    " have not been specified."
                )

        qubit_list = [
            qubit
            for qubit_list in server_qubits.values()
            for qubit in qubit_list
        ]
        if not len(qubit_list) == len(set(qubit_list)):
            raise Exception(
                "Qubits may belong to only one server"
                ", and should feature only once."
            )

        self.server_qubits = server_qubits

    def get_server_qubits(self):
        return self.server_qubits

    def get_nisq_nx(self) -> nx.Graph:

        G = nx.Graph()

        for qubits in self.server_qubits.values():
            for qubit_connection in combinations(qubits, 2):
                G.add_edge(*qubit_connection, color="r")

        for edge in self.server_coupling:
            G.add_edge(
                self.server_qubits[edge[0]][0],
                self.server_qubits[edge[1]][0],
                color="b",
            )

        return G

    def draw_nisq_network(self):

        G = self.get_nisq_nx()
        colors = [G[u][v]["color"] for u, v in G.edges()]
        nx.draw(G, with_labels=True, edge_color=colors)
