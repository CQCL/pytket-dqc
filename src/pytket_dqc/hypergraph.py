import hypernetx as hnx

class Hypergraph:
    
    def __init__(self):
        self.hyperedge_list = []
        self.vertex_list = []

    def draw(self):

        scenes = {}

        for i, edge in enumerate(self.hyperedge_list):
            scenes[str(i)] = set(edge)

        H = hnx.Hypergraph(scenes)
        hnx.drawing.draw(H)

    def add_vertex(self, vertex):
        if not vertex in self.vertex_list:
            self.vertex_list.append(vertex)

    def add_hyperedge(self, hyperedge: list[int]):

        for vertex in hyperedge:
            if not (vertex in self.vertex_list):
                raise Exception(
                    "An element of the hyperedge {} is not a vertex in {}. Please add it using add_vertex first".format(
                        hyperedge, self.vertex_list
                    )
                )

        self.hyperedge_list.append(hyperedge)