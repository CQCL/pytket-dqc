from __future__ import annotations

import networkx as nx  # type: ignore
from pytket_dqc.refiners import Refiner
from pytket_dqc.packing import PacMan, Packet
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

    def __init__(self):
        pass

    def refine(self, distribution: Distribution, **kwargs):
        """Updates the given distribution with a chosen allocation of qubits
        to choose a gate placement that minimises ebit count. Any prior gate
        placement is ignored.

        :param distribution: The distribution to be updated
        :type distribution: Distribution

        :key vertex_cover_alg: The choice of algorithm to be used to find the
        vertex covers that decide the placement of gates. Either:
            "all_brute_force" -> exhaustive search of all min vertex covers
            "networkx" -> use NetworkX's approximate algorithm to find a
                          single vertex cover
        """
        vertex_cover_alg = kwargs.get("vertex_cover_alg", None)
        if vertex_cover_alg is None or vertex_cover_alg not in [
            "all_brute_force",
            "networkx",
        ]:
            raise Exception(
                f"You must provide a vertex_cover_alg. Either:\n"
                + "\t\t\"all_brute_force\" -> "
                + "exhaustive search of all minimum vertex covers\n"
                + "\t\t\"networkx\" -> "
                + "use NetworkX's approximate algorithm to find a single vertex cover\n"
            )

        if vertex_cover_alg == "all_brute_force":
            self.exhaustive_refine(distribution)
        elif vertex_cover_alg == "networkx":
            self.networkx_refine(distribution)

    def exhaustive_refine(self, distribution: Distribution):
        """Refinement where all minimum vertex covers are found exhaustively.

        :param distribution: The distribution to be updated
        :type distribution: Distribution
        """
        pacman = PacMan(distribution.circuit, distribution.placement)
        merged_graph, _ = pacman.get_nx_graph_merged()
        conflict_graph, _ = pacman.get_nx_graph_conflict()

        # Find the vertex covers of each connected component separately
        full_valid_cover: list[tuple[Packet, ...]] = []
        for subgraph in [
            merged_graph.subgraph(c)
            for c in nx.connected_components(merged_graph)
        ]:

            # Step 1. Find all minimum vertex coverings of subgraph
            min_covers = get_min_covers(list(subgraph.edges))

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

        # Obtain a fresh HypergraphCircuit where no hyperedges are merged
        new_hyp_circ = pacman.get_hypergraph_from_packets()
        # Then, merge as required by ``full_valid_cover``
        for merged_packet in full_valid_cover:
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
        # Place gate vertices according to the packets in ``full_valid_cover``
        for merged_packet in full_valid_cover:
            for packet in merged_packet:
                server = packet.connected_server_index
                for vertex in packet.gate_vertices:
                    if vertex in new_placement.keys():
                        assert server == new_placement[vertex]
                    else:
                        new_placement[vertex] = server
        # Update the placement in ``distribution``
        distribution.placement = Placement(new_placement)

        # Sanity check
        distribution.is_valid()

    def networkx_refine(self, distribution: Distribution):
        """Refinement where the vertex covers are found using NetworkX's
        approximate algorithm. Only one vertex cover is found.

        :param distribution: The distribution to be updated
        :type distribution: Distribution
        """
        pacman = PacMan(distribution.circuit, distribution.placement)

        raise Exception("Not implemented.")


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
            for c in covers_w_v0: c.add(v0)
            # Omit all edges covered by v1 in recursive call
            covers_w_v1 = get_covers(
                [e for e in edges if e[0] != v1 and e[1] != v1]
            )
            for c in covers_w_v1: c.add(v1)
            # Return the union
            # NOTE: I'm not using set[set[Any]] instead of list[set[Any]]
            # due to set not being hashable -> I'd need set[frozenset[Any]]
            # but then it'd be a mess of types and I'd rather do this
            return covers_w_v0 + [c for c in covers_w_v1 if c not in covers_w_v0]

    # Filter out the covers that are not optimal
    covers = get_covers(edges)
    min_cover_size = min([len(c) for c in covers])
    min_covers = [c for c in covers if len(c) == min_cover_size]
    # There is always at least one cover.
    # Even in the case of no edges, we get the empty cover
    assert len(min_covers) > 0
    return min_covers