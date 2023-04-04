from __future__ import annotations

import networkx as nx  # type: ignore
from networkx.algorithms import bipartite  # type: ignore
from pytket_dqc.refiners import Refiner
from pytket_dqc.packing import PacMan, MergedPacket
from pytket_dqc.placement import Placement
from pytket_dqc.circuits import Hyperedge

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pytket_dqc import Distribution


class VertexCover(Refiner):
    """Refiner that leaves qubit allocation unchanged and decides gate
    placement from scratch. It uses a vertex cover approach compatible
    with embeddings and that prevents H-embedding conflicts.

    NOTE: Any prior gate placement is ignored.
    """

    def refine(self, distribution: Distribution, **kwargs) -> bool:
        """Updates the given distribution with a chosen allocation of qubits
        to choose a gate placement that minimises ebit count. Any prior gate
        placement is ignored.

        :param distribution: The distribution to be updated
        :type distribution: Distribution

        :key vertex_cover_alg: The choice of algorithm to be used to find the
        vertex covers that decide the placement of gates. Either:
        "all_brute_force" to do an exhaustive search of all min vertex covers
        or "networkx" to use NetworkX's algorithm to find a single cover.
        """
        vertex_cover_alg = kwargs.get("vertex_cover_alg", "networkx")
        if vertex_cover_alg not in [
            "all_brute_force",
            "networkx",
        ]:
            raise Exception(
                "You must provide a vertex_cover_alg. Either:\n"
                + "\t\t\"all_brute_force\" -> "
                + "exhaustive search of all minimum vertex covers\n"
                + "\t\t\"networkx\" -> "
                + "use NetworkX's algorithm to find a vertex cover\n"
            )

        pacman = PacMan(distribution.circuit, distribution.placement)
        # Decide on a cover using either approach
        if vertex_cover_alg == "all_brute_force":
            cover = self.exhaustive_refine(distribution, pacman)
        elif vertex_cover_alg == "networkx":
            cover = self.networkx_refine(distribution, pacman)

        # Obtain a fresh HypergraphCircuit where no hyperedges are merged
        new_hyp_circ = pacman.get_hypergraph_from_packets()
        # Then, merge as required by ``cover``
        for merged_packet in cover:
            # Skip if empty
            if len(merged_packet) == 0:
                continue
            # Otherwise gather all hyperedges
            qubit_vertex = merged_packet[0].qubit_vertex
            hyperedges: list[Hyperedge] = []
            for packet in merged_packet:
                assert qubit_vertex == packet.qubit_vertex
                hyperedges.append(
                    Hyperedge([qubit_vertex] + packet.gate_vertices)
                )
            # And merge them
            new_hyp_circ.merge_hyperedge(hyperedges)
        # Update the hypergraph in ``distribution``
        distribution.circuit = new_hyp_circ

        # Create a fresh placement dict with all qubits placed as originally
        new_placement = {
            q: distribution.placement.placement[q]
            for q in distribution.circuit.get_qubit_vertices()
        }
        # Place gate vertices according to the packets in ``cover``
        for merged_packet in cover:
            for packet in merged_packet:
                server = packet.connected_server_index
                for vertex in packet.gate_vertices:
                    new_placement[vertex] = server
        # Local gates are not considered above because packets for these
        # are not considered by `PacMan`. However, they must also be placed
        for vertex in new_hyp_circ.vertex_list:
            if vertex not in new_placement.keys():
                gate = new_hyp_circ.get_gate_of_vertex(vertex)
                q_vertices = [
                    new_hyp_circ.get_vertex_of_qubit(q) for q in gate.qubits
                ]

                # Sanity check: it is a local gate
                assert (
                    new_placement[q_vertices[0]]
                    == new_placement[q_vertices[1]]
                )
                # Place it in the local server
                new_placement[vertex] = new_placement[q_vertices[0]]

        # Update the placement in ``distribution``
        distribution.placement = Placement(new_placement)

        assert distribution.is_valid()
        return True

    def exhaustive_refine(
        self, distribution: Distribution, pacman: PacMan
    ) -> list[MergedPacket]:
        """Refinement where all minimum vertex covers are found exhaustively.

        :param distribution: The distribution to be updated
        :type distribution: Distribution
        :param pacman: The packet manager used during refinement
        :type pacman: PacMan
        :return: The list of selected merged packets to implement
        :rtype: list[MergedPacket]
        """
        merged_graph, _ = pacman.get_nx_graph_merged()
        conflict_graph, _ = pacman.get_nx_graph_conflict()

        # Find the vertex covers of each connected component separately
        full_valid_cover: list[MergedPacket] = []
        for subgraph in [
            merged_graph.subgraph(c)
            for c in nx.connected_components(merged_graph)
        ]:

            # Step 1. Find all minimum vertex coverings of subgraph
            min_covers: list[set[MergedPacket]] = get_min_covers(
                list(subgraph.edges)
            )

            # Find the best way to remove conflicts for each cover
            best_cover = None
            best_conflict_removal = None
            for cover in min_covers:
                # Step 2. Find all the hopping packets in ``cover``
                hop_packets = pacman.get_hopping_packets_within(cover)
                # Step 3. Find the best way to remove conflicts on ``cover``
                true_conflicts_g = conflict_graph.subgraph(hop_packets)
                conflict_covers = get_min_covers(list(true_conflicts_g.edges))
                # Pick one of the conflict_covers, all are optimal; there's
                # always at least one
                assert len(conflict_covers) > 0
                conflict_removal = conflict_covers[0]
                # Step 4. Find the best among cover after conflict removal
                # fmt: off
                if (
                    best_conflict_removal is None or
                    len(conflict_removal) < len(best_conflict_removal)
                ):
                    best_cover = cover
                    best_conflict_removal = conflict_removal
                # fmt: on
            assert best_cover is not None
            assert best_conflict_removal is not None

            # Step 5. Update ``best_cover`` by splitting according to
            # ``best_conflict_removal``
            for (p0, p1) in best_conflict_removal:
                # Retrieve the merged packet containing this conflict
                merged_packet = pacman.get_containing_merged_packet(p0)
                assert merged_packet == pacman.get_containing_merged_packet(p1)
                # Split the packet
                packet_a, packet_b = pacman.get_split_packets(
                    merged_packet, (p0, p1)
                )
                # Update the ``best_cover``
                best_cover.remove(merged_packet)
                best_cover.add(packet_a)
                best_cover.add(packet_b)

            # Include the cover of this subgraph in the full cover
            full_valid_cover += best_cover

        return full_valid_cover

    def networkx_refine(
        self, distribution: Distribution, pacman: PacMan
    ) -> list[MergedPacket]:
        """Refinement where the vertex covers are found using NetworkX's
        algorithm. Only one vertex cover is found.

        :param distribution: The distribution to be updated
        :type distribution: Distribution
        :param pacman: The packet manager used during refinement
        :type pacman: PacMan
        :return: The list of selected merged packets to implement
        :rtype: list[MergedPacket]
        """
        merged_graph, m_topnodes = pacman.get_nx_graph_merged()
        conflict_graph, c_topnodes = pacman.get_nx_graph_conflict()

        # Find a vertex cover
        matching = bipartite.maximum_matching(
            merged_graph, top_nodes=m_topnodes
        )
        cover = bipartite.to_vertex_cover(
            merged_graph, matching, top_nodes=m_topnodes
        )
        # Find all of the hopping packets in ``cover``
        hop_packets = pacman.get_hopping_packets_within(cover)

        # Find a way to remove the conflicts on ``cover``
        true_conflict_graph = conflict_graph.subgraph(hop_packets)
        tc_topnodes = {
            node for node in true_conflict_graph.nodes if node in c_topnodes
        }
        matching = bipartite.maximum_matching(
            true_conflict_graph, top_nodes=tc_topnodes
        )
        conflict_removal = bipartite.to_vertex_cover(
            true_conflict_graph, matching, top_nodes=tc_topnodes
        )

        # Update ``cover`` by splitting according to ``conflict_removal``
        for (p0, p1) in conflict_removal:
            # Retrieve the merged packet containing this conflict
            merged_packet = pacman.get_containing_merged_packet(p0)
            assert merged_packet == pacman.get_containing_merged_packet(p1)
            # Split the packet
            packet_a, packet_b = pacman.get_split_packets(
                merged_packet, (p0, p1)
            )
            # Update ``cover``
            cover.remove(merged_packet)
            cover.add(packet_a)
            cover.add(packet_b)

        return cover


def get_min_covers(edges: list[tuple[Any, Any]]) -> list[set[Any]]:
    """Recursive function that finds all minimum vertex covers of the given
    edges. Its complexity is at most O(2^c) where c is the size of the worst
    cover.
    """

    def get_covers(edges: list[tuple[Any, Any]]) -> list[set[Any]]:
        """All minimum vertex covers are guaranteed to be found by this
        recursive function, some extra non-minimal covers are also found.
        """
        if not edges:
            # Return a singleton list with the minimum cover: the trivial one
            return [set()]
        else:
            (v0, v1) = edges[0]
            # Omit all edges covered by v0 in recursive call
            covers_w_v0 = get_covers(
                [e for e in edges if e[0] != v0 and e[1] != v0]
            )
            for c in covers_w_v0:
                c.add(v0)
            # Omit all edges covered by v1 in recursive call
            covers_w_v1 = get_covers(
                [e for e in edges if e[0] != v1 and e[1] != v1]
            )
            for c in covers_w_v1:
                c.add(v1)
            # Return the union
            # NOTE: I'm not using set[set[Any]] instead of list[set[Any]]
            # due to set not being hashable -> I'd need set[frozenset[Any]]
            # but then it'd be a mess of types and I'd rather do this
            return covers_w_v0 + [
                c for c in covers_w_v1 if c not in covers_w_v0
            ]

    # Filter out the covers that are not optimal
    covers = get_covers(edges)
    min_cover_size = min([len(c) for c in covers])
    min_covers = [c for c in covers if len(c) == min_cover_size]
    # There is always at least one cover.
    # Even in the case of no edges, we get the empty cover
    assert len(min_covers) > 0
    return min_covers
