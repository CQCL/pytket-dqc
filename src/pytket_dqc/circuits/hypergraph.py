import hypernetx as hnx  # type: ignore


class Hypergraph:
    def __init__(self):
        self.hyperedge_list: list[list[int]] = []
        self.vertex_list: list[int] = []

    def is_placement(self, placement: dict[int, int]):

        valid = True

        # Check that every vertex is placed by this placement.
        for vertex in self.vertex_list:
            if vertex not in placement.keys():
                valid = False

        # Check that every vertex placed by this placement
        # is in the hypergraph.
        for vertex in placement.keys():
            if vertex not in self.vertex_list:
                valid = False

        return valid

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
