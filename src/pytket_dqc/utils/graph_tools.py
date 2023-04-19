# Copyright 2023 Quantinuum and The University of Tokyo
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import networkx as nx  # type: ignore
import networkx.algorithms.approximation.steinertree as st  # type: ignore
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


def steiner_tree(graph: nx.Graph, nodes: list[int]) -> nx.Graph:
    """Calls NetworkX's steiner_tree but manually manages the case of
    ``nodes`` being a singleton set, so that the singleton graph is
    returned, instead of the empty graph that NetworkX returns.
    """
    if len(nodes) == 0:
        raise Exception("No nodes have been provided")
    elif len(set(nodes)) == 1:
        tree = nx.Graph()
        tree.add_nodes_from(nodes)
        return tree
    else:
        return st.steiner_tree(graph, nodes)
