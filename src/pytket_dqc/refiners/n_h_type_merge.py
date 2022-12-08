from .refiner import Refiner
from pytket_dqc.circuits.distribution import Distribution
from pytket_dqc.allocators import GainManager
from pytket_dqc.circuits import Hyperedge
from pytket_dqc.packing import Packet, PacMan, HoppingPacket
from copy import copy


class NHTypeGreedyMerge(Refiner):
    """Scans circuit from left to right, merging hedges
    as it finds them according to `PacMan` found packets.
    """

    def refine(self, distribution: Distribution):
        gain_mgr = GainManager(initial_distribution=distribution)
        refinement_made = False
        pacman = PacMan(distribution.circuit, distribution.placement)

        already_done_hoppings: list[HoppingPacket] = list()
        all_hedges_to_merge: list[list[Hyperedge]] = list()
        currently_merging_hedges: set[Hyperedge] = set()

        # QUESTION: Is it preferable to call this via gain_mgr than just
        # through distribution.circuit directly?
        # (Currently this mirrors approaches in other refiners)
        for qubit_vertex in gain_mgr.distribution.circuit.get_qubit_vertices():
            qubit_packets = copy(pacman.packets_by_qubit[qubit_vertex])
            if qubit_packets:
                # Given a `Packet`, chain neighbouring and hoppable packets
                # together and add their parent hedges to a list of commonly
                # mergeable edge, removing the packets as we go.
                # If the chain stops then take the next packet in the list.
                #
                # Since packets by qubit is in circuit chronological order
                # (in theory by construction TODO: NOT verified)
                # then each all possible mergings should be done.
                current_packet: Packet = qubit_packets[
                    0
                ]  # Initial packet to start with
                end_merging_hedges = False
                while qubit_packets:  # Keep going until list is empty

                    if end_merging_hedges:
                        current_packet = qubit_packets[0]
                        # It might be the case that we have already
                        # considered the parent hedge but the derived
                        # `Packet`s were not mergeable
                        # This ensures that mutually mergeable hedges
                        # are put together
                        if any(
                            current_packet.parent_hedge in merging_hedges
                            for merging_hedges in all_hedges_to_merge[
                                qubit_vertex
                            ]
                        ):
                            currently_merging_hedges = set(
                                [
                                    merging_hedges
                                    for merging_hedges in all_hedges_to_merge
                                    if current_packet.parent_hedge
                                    in merging_hedges
                                ][0]
                            )

                        end_merging_hedges = False

                    currently_merging_hedges.add(current_packet.parent_hedge)
                    qubit_packets.remove(current_packet)
                    next_neighbour = pacman.get_subsequent_neighbouring_packet(
                        current_packet
                    )
                    next_hopper = pacman.get_subsequent_hopping_packet(
                        current_packet
                    )
                    if next_neighbour is not None:
                        current_packet = next_neighbour
                        currently_merging_hedges.add(
                            current_packet.parent_hedge
                        )
                    elif next_hopper is not None:
                        hopping_packet = (current_packet, next_hopper)
                        # We only merge embeddings when there are no conflicts
                        # that have already been merged previously
                        if all(
                            [
                                conflict not in already_done_hoppings
                                for conflict in pacman.get_conflict_hoppings(
                                    hopping_packet
                                )
                            ]
                        ):
                            already_done_hoppings.append(
                                hopping_packet
                            )  # Make a note that the hopping has now been done
                            current_packet = next_hopper
                        else:
                            end_merging_hedges = True
                    else:
                        end_merging_hedges = True

                    # End the merging and make a new one
                    if end_merging_hedges:
                        all_hedges_to_merge.append(
                            list(currently_merging_hedges)
                        )
                        currently_merging_hedges = set()

        for merging_hedges in all_hedges_to_merge:
            if len(merging_hedges) > 1:
                assert (gain_mgr.merge_hyperedge_gain(merging_hedges)) >= 0
                gain_mgr.merge_hyperedge(merging_hedges)
                refinement_made = True

        assert gain_mgr.distribution.is_valid()
        return refinement_made
