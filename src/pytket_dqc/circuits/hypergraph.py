from __future__ import annotations

import hypernetx as hnx  # type: ignore

from typing import TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    from pytket_dqc.placement import Placement


class Hypergraph:
    def __init__(self):
        self.hyperedge_list: list[list[int]] = []
        self.vertex_list: list[int] = []

    def is_placement(self, placement: Placement) -> bool:

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

    def draw(self):
        scenes = {}
        for i, edge in enumerate(self.hyperedge_list):
            scenes[str(i)] = set(edge)
        H = hnx.Hypergraph(scenes)
        hnx.drawing.draw(H)

    def add_vertex(self, vertex: int):
        if vertex not in self.vertex_list:
            self.vertex_list.append(vertex)

    def add_vertices(self, vertices: list[int]):
        for vertex in vertices:
            self.add_vertex(vertex)

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

    def kahypar_hyperedges(self) -> Tuple[list[int], list[int]]:

        hyperedges = [
            vertex
            for hyperedge in self.hyperedge_list
            for vertex in hyperedge
        ]

        hyperedge_indices = [0]
        for hyperedge in self.hyperedge_list:
            hyperedge_indices.append(len(hyperedge) + hyperedge_indices[-1])

        return hyperedge_indices, hyperedges
