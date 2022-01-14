import networkx as nx  # type: ignore


class ServerNetwork:

    def __init__(self, server_coupling: list[list[int]]):

        for server_list in server_coupling:
            if not len(server_list) == 2:
                raise Exception(
                    "server_coupling should be a list of pairs of servers.")

        self.server_coupling = server_coupling

    def is_placement(self, placement: dict[int, int]) -> bool:

        valid = True

        # Check that every server vertices are placed onto is in this network.
        server_list = self.get_server_list()
        for server in placement.values():
            if server not in server_list:
                valid = False

        return valid

    def get_server_list(self) -> list[int]:

        # Servers are saved in server_coupling as edges. Extract vertices here.
        expanded_coupling = [
            server for coupling in self.server_coupling for server in coupling
        ]
        return list(set(expanded_coupling))

    def get_server_nx(self) -> nx.Graph:

        G = nx.Graph()
        for edge in self.server_coupling:
            G.add_edge(edge[0], edge[1])
        return G

    def draw_server_network(self):

        G = self.get_server_nx()
        nx.draw(G, with_labels=True)
