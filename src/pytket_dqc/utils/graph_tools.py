import networkx as nx  # type: ignore
from typing import List


def direct_from_origin(G: nx.Graph, origin: int) -> List[List[int]]:
    """Generates list of edges in the graph G. These edges are ordered and
    directed so as to expand from an origin node.

    :param G: The original undirected graph. Note that nodes of this graph
        should be integers. G should be a tree.
    :type G: nx.Graph
    :param origin: Node from which edges should spread.
    :type origin: int
    :raises Exception: Raised if the graph G is not a tree.
    :raises Exception: Raised if the graph G is not connected.
    :return: List of edges.
    :rtype: List[List[int]]
    """

    if len(G.nodes()) == 0:
        return []

    if not nx.is_tree(G):
        raise Exception("The graph to redirect must be a tree.")

    # TODO: It may be possible to define some reaonsable behaviour in the case
    # where the graph is not connected but just covering this base for now.
    if not nx.is_connected(G):
        raise Exception("The graph is not connected.")

    edge_list = []

    # Remove origin node to create selection of subgraphs, each of which
    # is connected.
    G_reduced = G.copy()
    G_reduced.remove_node(origin)

    # Iterate though each neighbour of the origin node and repeat the
    # same process on the connected component of the reduced graph to which
    # the neighbour belongs.
    # Sorting this iterable is not strictly necessary here, although it
    # makes behaviour more predictable (and so testing a little easier).
    # For our purposes this means that distributed gates
    # are added to the circuit in the order in which the server indices
    # increase.
    for n in sorted(G.neighbors(origin)):

        # Retrieve the connected component to which the neighbour belongs.
        c = nx.node_connected_component(G_reduced, n)
        # Continue exploring the graph if a leaf node is not reached.
        if not len(c) == 1:
            edge_list += direct_from_origin(G_reduced.subgraph(c), n)

        # add origin node and neighbour to top of edge list.
        edge_list.insert(0, [origin, n])

    return edge_list
