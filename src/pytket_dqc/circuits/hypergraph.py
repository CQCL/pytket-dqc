import hypernetx as hnx  # type: ignore
from pytket_dqc.networks import NISQNetwork
import networkx as nx  # type: ignore


class Hypergraph:
    def __init__(self):
        self.hyperedge_list: list[list[int]] = []
        self.vertex_list: list[int] = []

    def is_placement(self, placement: dict[int, int], network: NISQNetwork):

        valid = True

        for vertex in self.vertex_list:
            if vertex not in placement.keys():
                valid = False

        for vertex in placement.keys():
            if vertex not in self.vertex_list:
                valid = False

        server_list = network.get_server_list()
        for server in placement.values():
            if server not in server_list:
                valid = False

        return valid

    def placement_cost(
        self,
        placement: dict[int, int],
        network: NISQNetwork
    ) -> int:

        cost = 0
        if self.is_placement(placement, network):

            G = network.get_server_nx()

            for hyperedge in self.hyperedge_list:
                hyperedge_placement = [
                    placement[vertex] for vertex in hyperedge
                ]
                unique_servers_used = list(set(hyperedge_placement))
                # TODO: This approach very naively assumes that the control is
                # teleported back when a new server pair is interacted.
                for server in unique_servers_used[1:]:
                    cost += nx.shortest_path_length(
                        G, unique_servers_used[0], server
                    )

        else:
            raise Exception("This is not a valid placement.")

        return cost

    def draw(self):
        scenes = {}
        for i, edge in enumerate(self.hyperedge_list):
            scenes[str(i)] = set(edge)
        H = hnx.Hypergraph(scenes)
        hnx.drawing.draw(H)

    def get_hyperedge_list(self):
        return self.hyperedge_list

    def get_vertex_list(self):
        return self.vertex_list

    def add_vertex(self, vertex: int):
        if vertex not in self.vertex_list:
            self.vertex_list.append(vertex)

    def add_hyperedge(self, hyperedge: list[int]):

        for vertex in hyperedge:
            if vertex not in self.vertex_list:
                raise Exception(
                    (
                        "An element of the hyperedge {} is not a vertex in {}."
                        "Please add it using add_vertex first"
                    ).format(hyperedge, self.vertex_list)
                )

        self.hyperedge_list.append(hyperedge)
