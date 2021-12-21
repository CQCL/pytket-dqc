import hypernetx as hnx  # type: ignore
from pytket_dqc.networks import NISQNetwork


class Hypergraph:
    def __init__(self):
        self.hyperedge_list: list[list[int]] = []
        self.vertex_list: list[int] = []

    def placement_cost(self, placement: dict[int, int], network: NISQNetwork):
        cost = 0
        for hyperedge in self.hyperedge_list:
            hyperedge_placement = [placement[vertex] for vertex in hyperedge]
            print("hyperedge_placement", hyperedge_placement)
            cost += len(set(hyperedge_placement)) - 1
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
