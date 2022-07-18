from __future__ import annotations

import hypernetx as hnx  # type: ignore
from collections import deque

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

    def __str__(self):
        out_string = f"Hyperedges: {self.hyperedge_list}"
        out_string += f"\nVertices: {self.vertex_list}"
        return out_string

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

    def add_hyperedge(self, vertices: list[Vertex], weight: int = 1):
        """Add hyperedge to hypergraph. Update vertex_neighbours.

        :param vertices: List of vertices in hyperedge
        :type vertices: list[Vertex]
        :param weight: Hyperedge weight
        :type weight: int
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

        self.hyperedge_list.append(hyperedge)

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


HedgeID = int


class CoarsenedPair(NamedTuple):
    """This is meant to be the type of the elements in the ``memento`` stack
    used when coarsening and uncoarsening a ``CoarseHyp``.

    :param representative: The vertex that acts as the representative of
        the pair after coarsening
    :type representative: Vertex
    :param hidden: The vertex that is hidden after coarsening
    :type hidden: Vertex
    :param relinked: The list of hyperedges incident to ``hidden`` that
        were relinked to ``representative`` while coarsening
    """
    representative: Vertex
    hidden: Vertex
    relinked: list[HedgeID]


class CoarseHyp():
    """TODO

    :param vertex_list: List of all vertices (including hidden ones)
    :type vertex_list: list[Vertex]
    :param qubit_vertices: The subset of vertices that correspond to qubits
    :type qubit_vertices: frozenset[Vertex]
    :param hidden_vertices: Set of hidden vertices; using set for efficiency
    :type hidden_vertices: set[Vertex]
    :param hyperedge_hash: Maps unique HedgeIDs to hyperedges. This is
        essential for fast hyperedge relinking (when un/coarsening)
    :type hyperedge_hash: dict[HedgeID, Hyperedge]
    :param hyperedge_dict: Maps each vertex to its incident hyperedges
    :type hyperedge_dict: dict[Vertex, list[HedgeID]]
    :param original_neighbours: Maps each vertex to its neighbourhood in the
        original hypergraph (uncoarsened)
    :type original_neighbours: dict[Vertex, set[Vertex]]
    :param memento: A stack of coarsened pairs, used to know how to uncoarsen
    :type memento: deque[Tuple[Vertex,Vertex]]
    """

    def __init__(self, hyp: Hypergraph, qubit_vertices: list[Vertex]):
        """Create a shallow copy of ``hyp``.
        """
        self.vertex_list: list[Vertex] = hyp.vertex_list
        self.qubit_vertices: frozenset[Vertex] = frozenset(qubit_vertices)
        self.hidden_vertices: set[Vertex] = set()
        self.hyperedge_hash: dict[HedgeID, Hyperedge] = dict()
        self.hyperedge_dict: dict[Vertex, set[HedgeID]] = dict()
        for vertex in hyp.vertex_list:
            self.hyperedge_dict[vertex] = set()
        for i, hedge in enumerate(hyp.hyperedge_list):
            self.hyperedge_hash[i] = hedge
            for vertex in hedge.vertices:
                self.hyperedge_dict[vertex].add(i)
        self.original_neighbours: dict[Vertex, set[Vertex]] = hyp.vertex_neighbours
        self.memento: deque[CoarsenedPair] = deque()

    def coarsen(self, rep: Vertex, to_hide: Vertex):
        """Coarsen the ``to_hide`` vertex into ``rep`` vertex adapting
        Algorithm 4.1 from Sebastian's thesis (KaHyPar). Coarsening means that
        the two vertices are merged into one, with ``rep`` representing them
        and the hyperedge adjacency being updated accordingly.
        In our case, we assume that ``rep`` is a qubit vertex and ``to_hide``
        is a gate vertex. We also assume that ``rep`` and ``to_hide`` are
        adjacent.
        """
        assert rep not in self.hidden_vertices
        assert to_hide not in self.hidden_vertices

        # The weight of the vertices does not need to be updated since
        # ``to_hide`` is assumed to be a gate vertex and, hence, has
        # weight 0

        # Iterate over the hyperedge incident to ``to_hide`` to update their
        # adjacency accordingly
        relinked = []
        for hedge_id in self.hyperedge_dict[to_hide]:
            hypedge = self.hyperedge_hash[hedge_id]
            # Remove ``to_hide`` and make sure the hyperedge is connected to
            # the new vertex (rep, to_hide) represented by ``rep``
            hypedge.vertices.remove(to_hide)
            # The dictionary entries of ``to_hide`` need not be removed nor
            # updated since we simply won't access them.
            if rep not in hypedge.vertices:
                # Relink operation; ``hypedge`` is no longer connected to the
                # new vertex (rep, to_hide). We need to fix this
                hypedge.vertices.append(rep)
                # We also need to add the hyperedge to entry of ``rep``
                self.hyperedge_dict[rep].append(hedge_id)
                # And we keep track of relinked hyperedges for later reference
                relinked.append(hedge_id)

        # Disable ``to_hide`` since it has been contracted into ``rep``
        self.hidden_vertices.add(to_hide)
        # Push the coarsened pair to the ``memento`` stack
        self.memento.append(CoarsenedPair(rep, to_hide, relinked))

    def uncoarsen(self) -> Tuple[Vertex, Vertex]:
        """Pop a CoarsenedPair (rep, hidden, relinked) from ``memento``
        then uncoarsen the ``hidden`` vertex from ``rep`` vertex adapting
        Algorithm 4.2 from Sebastian's thesis (KaHyPar). Uncoarsening should
        apply the inverse of coarsening.
        We assume that ``rep`` is a qubit vertex and ``hidden`` is a gate
        vertex.

        :return: The pair (rep, hidden).
        :rtype: Tuple[Vertex, Vertex]
        """
        assert self.memento
        coarsened_pair = self.memento.pop()

        self.hidden_vertices.remove(coarsened_pair.hidden)
        # The weight of the vertices does not need to be updated since
        # ``hidden`` is assumed to be a gate vertex and, hence,
        # has weight 0

        # We iterate over the references to the hyperedges that ``hidden``
        # was connected to before it was hidden. This works because
        # coarsening left the ``hyperedge_dict`` entry of ``to_hide``
        # unaltered and because the dict stores references to the Hyperedge
        # object, rather than its data.
        for hedge_id in self.hyperedge_dict[coarsened_pair.hidden]:
            hypedge = self.hyperedge_hash[hedge_id]
            # We need to restore ``to_extract`` into ``hypedge``
            hypedge.vertices.append(coarsened_pair.hidden)
        # We iterate over the relinked hyperedges and remove ``rep`` from them
        # thus reverting the relinking done during coarsening
        for hedge_id in coarsened_pair.relinked:
            hypedge = self.hyperedge_hash[hedge_id]
            hypedge.vertices.remove(coarsened_pair.representative)
            # We also need to remove the hyperedge from the entry ``rep``.
            # Doing so would be very costly if it were a set(Hyperedge)
            # instead of a set(HedgeID).
            self.hyperedge_dict[coarsened_pair.representative].remove(hedge_id)

        return coarsened_pair.representative, coarsened_pair.hidden

    def current_neighbours(self, vertex: Vertex) -> set[Vertex]:
        """Returns the neighbourhood of ``vertex`` in the coarsened hypergraph
        """
        assert vertex not in self.hidden_vertices
        # Notice that the data structure does not keep track of neighbourhoods
        # as the hypergraph is coarsened. This is because figuring out how
        # to restore the neighbourhoods after uncoarsening would be tough.
        # In particular, deciding whether reverting a relink should remove
        # ``rep`` from a neighbourhood cannot be decided without exploring
        # the hypergraph or adding more information to ``memento``. Thus,
        # the approach used here is to compute the current neighbourhood as
        # a view of the uncoarsened neighbourhood via ``memento``.
        uncoarsened = self.original_neighbours[vertex]
        neighbourhood = set(uncoarsened)
        # Read ``memento`` from the first entry to the last (chronologically)
        # so that we can reproduce the effect of coarsening in neighbourhood
        for cp in self.memento:
            if cp.hidden in neighbourhood:
                # Remove the hidden vertex from the neighbourhood
                neighbourhood.remove(cp.hidden)
                # In case of a relink, we need to add the representative to
                # the neighbourhood. If there was no relink it means the
                # representative vertex already was in the neighbourhood
                # so adding it to the set leaves it unchanged.
                neighbourhood.add(cp.representative)

        # When relinking it is possible that ``vertex`` itself was added to
        # its own neighbourhood. Thus, we remove it if present.
        if vertex in neighbourhood: neighbourhood.remove(vertex)
        return neighbourhood

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

            for neighbour in self.current_neighbours(vertex):
                if my_block != placement.placement[neighbour]:
                    boundary.append(vertex)
                    break

        return boundary

    def to_static_hypergraph(self) -> Hypergraph:
        """Return an instance of Hypergraph that is a copy of the current
        coarsened hypergraph. This function is expensive and should be
        seldomly called.
        """
        hypergraph = Hypergraph()

        # Vertices ought to be a continuous sequence of integers starting at 0
        # but the set of non-hidden vertices need not satisfy this. Hence, we
        # need to relabel them
        relabelling = dict()
        i = 0
        for vertex in self.vertex_list:
            # Assuming qubit vertices are at the beginning of the list, the
            # new vertex list will also satisfy this
            if vertex not in self.hidden_vertices:
                relabelling[i] = vertex
                hypergraph.add_vertex(i)
                i += 1
        # Add each hyperedge after relabelling its vertices
        for hyperedge in self.hyperedge_hash.values():
            vertices = [relabelling[v] for v in hyperedge.vertices]
            hypergraph.add_hyperedge(vertices, hyperedge.weight)

        assert hypergraph.is_valid()
        return hypergraph

    def kahypar_hyperedges(self) -> Tuple[list[int], list[int]]:
        """Return hypergraph in format used by kahypar package. In particular
        a list of vertices ``hyperedges``, and a list ``hyperedge_indices`` of
        indices of ``hyperedges``. ``hyperedge_indices`` gives a list of
        intervals, where each interval defines a hyperedge.

        :return: Hypergraph in format used by kahypar package.
        :rtype: Tuple[list[int], list[int]]
        """
        return self.to_static_hypergraph().kahypar_hyperedges()
