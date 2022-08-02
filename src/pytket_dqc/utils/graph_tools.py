import networkx as nx  # type: ignore
import networkx.algorithms.approximation.steinertree as st  # type: ignore
from networkx.algorithms import descendants
from itertools import combinations
from typing import List, Tuple


def direct_from_origin(G: nx.Graph, origin: int) -> List[Tuple[int, int]]:
    """Generates list of edges in the tree G. These edges are ordered and
    directed so as to expand from an origin node.

    :param G: The original undirected graph. Note that nodes of this graph
        should be integers. G should be a tree.
    :type G: nx.Graph
    :param origin: Node from which edges should spread.
    :type origin: int
    :raises Exception: Raised if the graph G is not a tree.
    :raises Exception: Raised if the graph G is not connected.
    :return: List of edges.
    :rtype: List[Tuple[int]]
    """

    if len(G.nodes()) == 0:
        return []

    if not nx.is_tree(G):
        raise Exception("The graph to redirect must be a tree.")

    # TODO: It may be possible to define some reaonsable behaviour in the case
    # where the graph is not connected but just covering this base for now.
    if not nx.is_connected(G):
        raise Exception("The graph is not connected.")

    # Remove origin node to create selection of subgraphs, each of which
    # is connected.
    G_reduced = G.copy()
    G_reduced.remove_node(origin)

    # add origin node and neighbour to top of edge list.
    edge_list = [(origin, n) for n in sorted(G.neighbors(origin))]

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

    return edge_list


def steiner_tree(
    graph: nx.Graph, terminal_nodes: list[int], brute_force: bool
) -> nx.Graph:
    """If ``brute_force`` is True, compute the optimal Steiner tree by brute
    force search. Otherwise, use the algorithm from NetworkX.
    """

    # Run the algorithm from NetworkX; if ``brute_force`` is enabled, its
    # solution will be used as the baseline
    tree = st.steiner_tree(graph, terminal_nodes)
    if not brute_force:
        return tree

    # If there are only two or fewer terminal nodes, NetworkX is guaranteed to
    # return the optimal Steiner tree (i.e. the shortest path, a single vertex
    # graph or an empty graph for 2, 1 or 0 terminal nodes, respectively)
    if len(terminal_nodes) <= 2:
        return tree

    # Some pre-processing that will be useful later
    source = terminal_nodes[0]
    other_terminal_nodes = set(terminal_nodes[1:])

    # Consider all subgraphs of cost lower than ``cost``. Do so iteratively by
    # first considering all subgraphs of ``cost-1`` and, if one is found,
    # repeat for ``cost-2``, etc.
    cost = len(tree.edges)
    valid = True
    while valid and cost > 1:
        valid = False
        # Consider all subgraphs with ``cost-1`` edges
        for edge_sublist in combinations(graph.edges, cost - 1):
            subgraph = nx.Graph(edge_sublist)
            # Check whether the subgraph is a valid Steiner tree
            if source in subgraph.nodes:
                # Obtain all vertices that are reachable from ``source``
                reachable = descendants(subgraph, source)
                # If all terminal nodes are reachable this is a Steiner tree
                if other_terminal_nodes.issubset(reachable):
                    valid = True
                    tree = subgraph
                    # Since we found a tree with ``cost-1`` edges we halt the
                    # for loop and do another iteration of the while loop
                    cost -= 1
                    break
    return tree
