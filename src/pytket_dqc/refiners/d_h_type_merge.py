from .refiner import Refiner
from pytket_dqc.circuits.distribution import Distribution
from pytket_dqc.allocators import GainManager
from pytket_dqc.circuits import Hyperedge
from pytket_dqc.packing import (
    PacMan,
    HoppingPacket
)


class DHTypeGreedyMerge(Refiner):
    """Scans circuit from left to right, merging hedges
    as it finds them according to `PacMan` found packets.
    """

    def refine(self, distribution: Distribution):
        gain_mgr = GainManager(initial_distribution=distribution)
        pacman = PacMan(distribution.circuit, distribution.placement)

        already_done_hoppings: list[HoppingPacket] = []
        all_hedges_to_merge: list[list[Hyperedge]] = []
        currently_merging_hedges: list[Hyperedge] = []

        # QUESTION: Is it preferable to call this via gain_mgr than just
        # through distribution.circuit directly?
        # (Currently this mirrors approaches in other refiners)
        for qubit_vertex in gain_mgr.distribution.circuit.get_qubit_vertices():
            packets = pacman.packets_by_qubit[qubit_vertex]
            for packet0, packet1 in zip(packets[:-1], packets[1:]):
                currently_merging_hedges.append(
                    packet0.packet_hedge
                )
                if pacman.are_neighbouring_packets(packet0, packet1):
                    # Can keep merging hedges
                    continue

                # If it's a hopping packet, then we need to check if any
                # of the contained embedded packets are connected to other
                # embedded packets.
                # If they are, then we need to check if those hopping packets
                # have already been implemented.
                # If so, then this hopping packet cannot be implemented.
                # NOTE: This is a bit inconsistent but I think faster than
                # running pacman.are_hoppable_packets() again?
                elif (packet0, packet1) in pacman.hopping_packets[
                    qubit_vertex
                ]:
                    conflict_hoppings = pacman.get_conflict_hoppings(
                        (packet0, packet1)
                    )

                    # Can't do the hopping
                    if any(
                        [
                            conflict_hopping in already_done_hoppings
                            for conflict_hopping in conflict_hoppings
                        ]
                    ):
                        all_hedges_to_merge.append(currently_merging_hedges)
                        currently_merging_hedges = []
                    else:
                        already_done_hoppings.append((packet0, packet1))

                else:
                    all_hedges_to_merge.append(currently_merging_hedges)
                    currently_merging_hedges = []
            # I have the same 3 pairs of lines here.
            # The logic can definitely be revisited to shorten this
            if currently_merging_hedges:
                all_hedges_to_merge.append(currently_merging_hedges)
                currently_merging_hedges = []

        refinement_made = False

        # Merge and store the grouped hedges
        for hedges_to_merge in all_hedges_to_merge:
            if not len(hedges_to_merge) == 1:
                assert (
                    gain_mgr.merge_hyperedge_gain(hedges_to_merge) > 0
                ), "The gain should be at least one..."
                gain_mgr.merge_hyperedge(hedges_to_merge)
                refinement_made = True

        assert gain_mgr.distribution.is_valid()
        return refinement_made
