from __future__ import annotations

import hypernetx as hnx  # type: ignore

from typing import TYPE_CHECKING, Tuple, NamedTuple

if TYPE_CHECKING:
    from pytket_dqc.placement import Placement

# Custom types
Vertex = int


class Hyperedge(NamedTuple):
    vertices: list[Vertex]
    weight: int = 1

    def __hash__(self):
        return hash((frozenset(self.vertices), self.weight))


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

    def __str__(self):
        out_string = f"Hyperedges: {self.hyperedge_list}"
        out_string += f"\nVertices: {self.vertex_list}"
        return out_string

    def merge_hyperedge(
        self,
        to_merge_hyperedge_list: list[Hyperedge]
    ) -> Hyperedge:
        """Merge vertices of each of the hyperedges in to_merge_hyperedge_list
        into a single hyperedge. The new hyperedge will appear in
        `hyperedge_list` at the lowest index of the hyperedges in
        `to_merge_hyperedge_list`.

        :param to_merge_hyperedge_list: List of hyperedges to merge.
        :type to_merge_hyperedge_list: list[Hyperedge]
        :raises Exception: Raised if any of the hyperedges in
        to_merge_hyperedge_list are not in this hypergraph.
        :raises Exception: Raised if the weights of the hyperedges to merge
        do not match.
        """

        if not all(
            to_merge_hyperedge in self.hyperedge_list
            for to_merge_hyperedge in to_merge_hyperedge_list
        ):
            raise Exception(
                "At least one hyperedge in to_merge_hyperedge_list " +
                "does not belong to this hypergraph."
            )

        if not all(
            to_merge_hyperedge_list[0].weight == to_merge_hyperedge.weight
            for to_merge_hyperedge in to_merge_hyperedge_list
        ):
            raise Exception("Weights of hyperedges to merge should be equal.")

        # Gather list of all vertices in hyperedges to be merged.
        vertices = list(
            set(
                vertex
                for to_merge_hyperedge in to_merge_hyperedge_list
                for vertex in to_merge_hyperedge.vertices
            )
        )
        weight = to_merge_hyperedge_list[0].weight
        index = min(self.hyperedge_list.index(hyperedge)
                    for hyperedge in to_merge_hyperedge_list)
        new_hyperedge = Hyperedge(vertices=vertices, weight=weight)
        self.add_hyperedge(vertices=new_hyperedge.vertices,
                           weight=new_hyperedge.weight, index=index)

        removed_hyperedge_list: list[Hyperedge] = []
        for hyperedge in to_merge_hyperedge_list:
            try:
                self.remove_hyperedge(hyperedge)
            except Exception:
                for removed_hyperedge in removed_hyperedge_list:
                    self.add_hyperedge(
                        vertices=removed_hyperedge.vertices,
                        weight=removed_hyperedge.weight
                    )
                self.remove_hyperedge(new_hyperedge)
                raise
            removed_hyperedge_list.append(hyperedge)

        return Hyperedge(vertices=vertices, weight=weight)

    def split_hyperedge(
        self,
        old_hyperedge: Hyperedge,
        new_hyperedge_list: list[Hyperedge]
    ):
        """Split `old_hyperedge` into the hyperedges in `new_hyperedge_list`.
        The new hyperedges will appear in `hyperedge_list` at the
        same location as the `old_hyperedge` in the same order as they
        appear in `new_hyperedge_list`.

        :param old_hyperedge: Hyperedge to split.
        :type old_hyperedge: Hyperedge
        :param new_hyperedge_list: List of hyperedges into which
        `old_hyperedge` should be split.
        :type new_hyperedge_list: list[Hyperedge]
        :raises Exception: Raised if `new_hyperedge_list` is not a valid
        split of `old_hyperedge`
        """

        flat_vertex_list = [
            vertex for hypergraph in new_hyperedge_list
            for vertex in hypergraph.vertices
        ]
        if not (set(flat_vertex_list) == set(old_hyperedge.vertices)):
            raise Exception(
                f"{new_hyperedge_list} does not " +
                f"match the vertices in {old_hyperedge}"
            )

        index = self.hyperedge_list.index(old_hyperedge)
        added_hyperedge_list: list[Hyperedge] = []
        for new_hyperedge in reversed(new_hyperedge_list):
            try:
                self.add_hyperedge(new_hyperedge.vertices,
                                   new_hyperedge.weight, index=index)
            except Exception:
                for hyperedge in added_hyperedge_list:
                    self.remove_hyperedge(hyperedge)
                raise
            added_hyperedge_list.append(new_hyperedge)
        self.remove_hyperedge(old_hyperedge)

    def remove_hyperedge(self, old_hyperedge: Hyperedge):
        """Remove hypergraph. Update vertex_neighbours.

        :param old_hyperedge: Hyperedge to remove
        :type vertices: Hyperedge
        :raises Exception: Raised if `old_hyperedge` is not in hypergraph.
        """

        # This might be a bit server is general, as we could just
        # ignore it if the hyperedge to remove is not present.
        # As the developers are the only users we might prefer to
        # know about this though.
        if old_hyperedge not in self.hyperedge_list:
            raise Exception(
                f"The hyperedge {old_hyperedge} is not in this hypergraph."
            )

        self.hyperedge_list.remove(old_hyperedge)
        # For every vertex in the hyperedge being removed, update
        # appropriately if it is still a neighbour to other vertices.
        for vertex in old_hyperedge.vertices:
            old_neighbour_list = set(old_hyperedge.vertices) - {vertex}
            # For every old_neighbour of vertex, check if the pair both
            # belong to another hyperedge. Update vertex_neighbours
            # accordingly.
            for old_neighbour in old_neighbour_list:
                paired_elsewhere = any(
                    {vertex, old_neighbour} <= set(hyperedge.vertices)
                    for hyperedge in self.hyperedge_list
                )
                if not paired_elsewhere:
                    self.vertex_neighbours[vertex].discard(old_neighbour)
                    self.vertex_neighbours[old_neighbour].discard(vertex)

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

        # The following assert is guaranteed by construction
        assert sorted(self.vertex_list) == sorted(
            list(self.vertex_neighbours.keys())
        )

        vertex_list_sorted = self.vertex_list.copy()
        vertex_list_sorted.sort()
        unique_vertex_list_sorted = list(set(vertex_list_sorted))

        if len(self.vertex_list) == 0:
            ideal_vertex_list = []
        else:
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

    def add_hyperedge(
        self,
        vertices: list[Vertex],
        weight: int = 1,
        index: int = None
    ):
        """Add hyperedge to hypergraph. Update vertex_neighbours.

        :param vertices: List of vertices in hyperedge
        :type vertices: list[Vertex]
        :param weight: Hyperedge weight
        :type weight: int
        :param index: index in `hyperedge_list` at which the new
        hyperedge will be added.
        :type index: int
        :raises Exception: Raised if hyperedge does not contain at least
            2 vertices
        :raises Exception: Raised if vertices in hyperedge are not in
            hypergraph.
        """

        if len(vertices) < 1:
            raise Exception("Hyperedges must contain at least 1 vertex.")

        hyperedge = Hyperedge(vertices, weight)
        for vertex in vertices:
            if vertex not in self.vertex_list:
                raise Exception(
                    (
                        "An element of the hyperedge {} is not a vertex in {}."
                        "Please add it using add_vertex first"
                    ).format(hyperedge, self.vertex_list)
                )

            self.hyperedge_dict[vertex].append(hyperedge)

            # Add in all vertices of the hyperedge to the neighbourhood. Since
            # this is a set there will be no duplicates. This carelessly adds
            # in the vertex itself to its own neighbourhood, so we remove it.
            self.vertex_neighbours[vertex].update(vertices)
            self.vertex_neighbours[vertex].remove(vertex)

        if index is None:
            self.hyperedge_list.append(hyperedge)
        else:
            self.hyperedge_list.insert(index, hyperedge)

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

        if len(self.hyperedge_list) == 0:
            return [], hyperedges

        hyperedge_indices = [0]
        for hyperedge in self.hyperedge_list:
            hyperedge_indices.append(
                len(hyperedge.vertices) + hyperedge_indices[-1]
            )

        return hyperedge_indices, hyperedges

    def get_boundary(self, placement: Placement) -> list[Vertex]:
        """Given a placement of vertices to blocks, find the subset of vertices
        in their boundaries. A boundary vertex is a vertex in some block B1
        that has a neighbour in another block B2.

        :param placement: An assignment of vertices to blocks
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

    def weight_one_predicate(self) -> bool:
        """Returns True if all of the hyperedges in the hypergraph have weight
        equal to one.
        """
        for hyperedge in self.hyperedge_list:
            if hyperedge.weight != 1:
                return False

        return True
