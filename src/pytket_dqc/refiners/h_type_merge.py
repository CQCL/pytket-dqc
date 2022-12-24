from .refiner import Refiner
from pytket_dqc.circuits.distribution import Distribution
from pytket_dqc.allocators import GainManager
from pytket_dqc.circuits import Hyperedge
from pytket_dqc.packing import Packet, PacMan, HoppingPacket
from copy import copy


class EagerHTypeMerge(Refiner):
    """Scans circuit from left to right, merging hedges
    via hoppings as it finds them according to `PacMan` found packets.

    In the case of conflicts, the first found hopping is the one that is
    used, and no calculation is made as to which might be better to implement.
    """

    def refine(self, distribution: Distribution):
        gain_mgr = GainManager(initial_distribution=distribution)
        refinement_made = False
        pacman = PacMan(distribution.circuit, distribution.placement)

        already_done_hoppings: list[HoppingPacket] = list()
        all_hedges_to_merge: list[set[Hyperedge]] = list()
        currently_merging_hedges: set[Hyperedge] = set()

        for qubit_vertex in gain_mgr.distribution.circuit.get_qubit_vertices():
            qubit_packets = copy(pacman.packets_by_qubit[qubit_vertex])
            if qubit_packets:
                # Given a `Packet`, chain hoppable `Packet`s
                # together and add their parent hedges to a list of commonly
                # mergeable `Hyperedge`s, removing the `Packet`s as we go.
                # If the chain stops then take the next `Packet`` in the list.
                #
                # `PacMan.packets_by_qubit[qubit_vertex]` generates a list of
                # `Packet`s such that `Packet`s which are connected to the same
                # server are in chronological order.
                # Therefore starting the chain from the first `Packet`
                # in the list will find all the `Packet`s that can be
                # merged with the first since `Packet` mergings aren't
                # allowed for `Packet`s connected to different servers.

                # Initial `Packet` to start with
                current_packet: Packet = qubit_packets[
                    0
                ]
                currently_merging_hedges = {current_packet.parent_hedge}
                all_hedges_to_merge.append(currently_merging_hedges)
                end_merging_hedges = False
                while qubit_packets:  # Keep going until list is empty
                    qubit_packets.remove(current_packet)

                    # Identify the next mergeable `Packet` via hopping
                    next_hopper = pacman.get_subsequent_hopping_packet(
                        current_packet
                    )
                    if next_hopper is not None:
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
                            # Make a note that the hopping has now been done
                            # therefore we cannot do another hopping that
                            # conflicts with this one.
                            already_done_hoppings.append(
                                hopping_packet
                            )
                            current_packet = next_hopper
                            currently_merging_hedges.add(
                                current_packet.parent_hedge
                            )
                        else:
                            end_merging_hedges = True
                    else:
                        end_merging_hedges = True

                    # If no more `Hyperedge`s could be found to
                    # merge with, and there are still `Packet`s
                    # left to check, then start the search again
                    # from the next `Packet` in the list.
                    if end_merging_hedges and qubit_packets:
                        current_packet = qubit_packets[0]
                        # If the parent_hedge of this `Packet` is already
                        # part of a set of mergeable `Hyperedge`s,
                        # retrieve and append to that set
                        potential_merging_hedges_list = [
                            merging_hedges
                            for merging_hedges in all_hedges_to_merge
                            if current_packet.parent_hedge in merging_hedges
                        ]
                        assert len(potential_merging_hedges_list) <= 1,\
                            "There should only be up to one merging_hedges " +\
                            "for any hedge"
                        if potential_merging_hedges_list:
                            currently_merging_hedges \
                                = potential_merging_hedges_list[0]
                        else:
                            currently_merging_hedges = {
                                current_packet.parent_hedge
                            }
                            all_hedges_to_merge.append(
                                currently_merging_hedges
                            )
                        end_merging_hedges = False

        for merging_hedges in all_hedges_to_merge:
            if len(merging_hedges) > 1:
                # This merger resolves conflict edges by
                # construction - hence any merged hyperedges
                # should only require local correcting gates.
                # So the gain should be at least 1 if hyperedges
                # have been merged - it is only 0 in the case
                # that the placement does not make this merged
                # hyperedge non-local
                # (i.e all the gates are made local).
                assert (gain_mgr.merge_hyperedge_gain(
                    list(merging_hedges)
                )) >= 0
                gain_mgr.merge_hyperedge(list(merging_hedges))
                refinement_made = True

        assert gain_mgr.distribution.is_valid()
        return refinement_made
