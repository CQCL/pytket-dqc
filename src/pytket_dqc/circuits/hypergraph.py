from __future__ import annotations

import hypernetx as hnx  # type: ignore

from typing import TYPE_CHECKING, Tuple, NamedTuple

if TYPE_CHECKING:
    from pytket_dqc.placement import Placement

# Custom types
Vertex = int


class Hyperedge(NamedTuple):
    vertices: list[Vertex]
    weight: int


class Hypergraph:
    """A representation of a hypergraph. Hypergraphs are represented by
    vertices and hyperedges, where hyperedges consist of a collection of
    two or more vertices.
    Since neighbourhoods and incident hyperedge list will be often used
    it is best to store them in the data structure.

    :param vertex_list: List of vertices
    :type vertex_list: list[Vertex]
    :param hyperedge_list: List of hyperedges
    :type hyperedge_list: list[Hyperedge]
    :param hyperedge_dict: Maps each vertex to its incident hyperedges
    :type hyperedge_dict: dict[Vertex, list[Hyperedge]]
    :param vertex_neighbours: Maps each vertex to its neighbourhood
    :type vertex_neighbours: dict[Vertex, set[Vertex]]
    """

    def __init__(self):
        """Initialisation function. The hypergraph initialises as empty.
        """
        self.vertex_list: list[Vertex] = []
        self.hyperedge_list: list[Hyperedge] = []
        self.hyperedge_dict: dict[Vertex, list[Hyperedge]] = dict()
        self.vertex_neighbours: dict[Vertex, set[Vertex]] = dict()

    def is_placement(self, placement: Placement) -> bool:
        """Checks if a given placement is a valid placement of this hypergraph.
        Checks for example that all vertices are placed, and that every vertex
        placed is in the hypergraph.

        :param placement: The placement to check for validity
        :type placement: Placement
        :return: Is the placement valid.
        :rtype: bool
        """

        valid = True

        # Check that every vertex is placed by this placement.
        for vertex in self.vertex_list:
            if vertex not in placement.placement.keys():
                valid = False

        # Check that every vertex placed by this placement
        # is in the hypergraph.
        for vertex in placement.placement.keys():
            if vertex not in self.vertex_list:
                valid = False

        return valid

    # TODO: Is it possible to ensure this condition at the point of design
    # TODO: check that the hyperedges are unique
    # TODO: There are other checks to do like that each hyperedge has one
    # qubit and each gate is in at most two edges.
    def is_valid(self) -> bool:
        """Checks if this is a valid hypergraph. In particular vertices must
        be a continuous list of integers.

        :return: Is hypergraph valid.
        :rtype: bool
        """

        vertex_list_sorted = self.vertex_list.copy()
        vertex_list_sorted.sort()
        unique_vertex_list_sorted = list(set(vertex_list_sorted))

        ideal_vertex_list = [i for i in range(max(self.vertex_list) + 1)]

        # The vertices in the hypergraph must be a continuous list of integers.
        return unique_vertex_list_sorted == ideal_vertex_list

    def draw(self):
        """Draw hypergraph, using hypernetx package.
        """
        scenes = {}
        for i, edge in enumerate(self.hyperedge_list):
            scenes[str(i)] = set(edge.vertices)
        H = hnx.Hypergraph(scenes)
        hnx.drawing.draw(H)

    def add_vertex(self, vertex: Vertex):
        """Add vertex to hypergraph.

        :param vertex: Index of vertex.
        :type vertex: Vertex
        """
        if vertex not in self.vertex_list:
            self.vertex_list.append(vertex)
            self.hyperedge_dict[vertex] = []
            self.vertex_neighbours[vertex] = set()

    def add_vertices(self, vertices: list[Vertex]):
        """Add list ov vertices to hypergraph.

        :param vertices: List of vertex indices
        :type vertices: list[Vertex]
        """
        for vertex in vertices:
            self.add_vertex(vertex)

    def add_hyperedge(self, hyperedge: list[Vertex], weight: int = 1):
        """Add hyperedge to hypergraph. Update vertex_neighbours.

        :param hyperedge: List of vertices in hyperedge
        :type hyperedge: list[Vertex]
        :param weight: Hyperedge weight
        :type weight: int
        :raises Exception: Raised if hyperedge does not contain at least
            2 vetices
        :raises Exception: Raised if vertices in hyperedge are not in
            hypergraph.
        """

        if len(hyperedge) < 2:
            raise Exception("Hyperedges must contain at least 2 vertices.")

        for vertex in hyperedge:
            if (
                vertex not in self.vertex_list
                or vertex not in self.vertex_neighbours.keys()
            ):
                raise Exception(
                    (
                        "An element of the hyperedge {} is not a vertex in {}."
                        "Please add it using add_vertex first"
                    ).format(hyperedge, self.vertex_list)
                )

            self.hyperedge_dict[vertex].append(Hyperedge(hyperedge, weight))

            # Add in all vertices of the hyperedge to the neighbourhood. Since
            # this is a set there will be no duplicates. This carelessly adds
            # in the vertex itself to its own neighbourhood, so we remove it.
            self.vertex_neighbours[vertex].update(hyperedge)
            self.vertex_neighbours[vertex].remove(vertex)

        self.hyperedge_list.append(Hyperedge(hyperedge, weight))

    def kahypar_hyperedges(self) -> Tuple[list[int], list[int]]:
        """Return hypergraph in format used by kahypar package. In particular
        a list of vertices ``hyperedges``, and a list ``hyperedge_indices`` of
        indices of ``hyperedges``. ``hyperedge_indices`` gives a list of
        intervals, where each interval defines a hyperedge.

        :return: Hypergraph in format used by kahypar package.
        :rtype: Tuple[list[int], list[int]]
        """

        # List all hyper edges as continuous list of vertices.
        hyperedges = [
            vertex
            for hyperedge in self.hyperedge_list
            for vertex in hyperedge.vertices
        ]

        # Create list of intervals of hyperedges list which correspond to
        # hyperedges.
        hyperedge_indices = [0]
        for hyperedge in self.hyperedge_list:
            hyperedge_indices.append(
                len(hyperedge.vertices)
                + hyperedge_indices[-1]
            )

        return hyperedge_indices, hyperedges

    def get_boundary(self, placement: Placement) -> list[Vertex]:
        """Given a placement of vertices to blocks, find the subset of vertices
        in their boundaries. A boundary vertex is a vertex in some block B1
        that has a neighbour in another block B2.

        :param placement: An assignemnt of vertices to blocks
        :type placement: Placement

        :return: The list of boundary vertices
        :rtype: list[Vertex]
        """

        boundary = list()

        for vertex in self.vertex_list:
            my_block = placement.placement[vertex]

            for neighbour in self.vertex_neighbours[vertex]:
                if my_block != placement.placement[neighbour]:
                    boundary.append(vertex)
                    break

        return boundary
