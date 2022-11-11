from .refiner import Refiner
from pytket_dqc.circuits.distribution import Distribution
from pytket_dqc.allocators import GainManager
from pytket import OpType


class MergeDType(Refiner):
    """Refiner merging neighbouring packets when they do not require
    H-type embeddings.
    """

    def refine(self, distribution: Distribution):
        """Merges neighbouring packets when no H-type embeddings are
        required by either packet. Packets concerning a particular qubit are
        all merged where possible.

        :param distribution: Distribution whose packets should be
            merged.
        :type distribution: Distribution
        """

        gain_mgr = GainManager(initial_distribution=distribution)

        # Iterate through all qubits, merging packets.
        for qubit_vertex in gain_mgr.distribution.circuit.get_qubit_vertices():

            # List of hyperedges acting on qubit
            hedge_list = [
                hedge
                for hedge in gain_mgr.distribution.circuit.hyperedge_list
                if (
                    gain_mgr.distribution.circuit.get_qubit_vertex(hedge)
                    == qubit_vertex
                )
            ]

            # iterates through list of hyperedges. If the first element in
            # list can be merged with the next, do so, add the merged
            # hyperedge to the start of the list, and remove the two original
            # hyperedges. If it cannot be merged with the next, remove it
            # from the list. Repear until the list is empty.
            while len(hedge_list) >= 2:

                # Hyperedges to try to merge.
                hedge_one = hedge_list.pop(0)
                hedge_two = hedge_list[0]

                # commands between hyperedges.
                intermediate_commands = gain_mgr.distribution.circuit.get_intermediate_commands(  # noqa: E501
                    first_vertex=max(
                        gain_mgr.distribution.circuit.get_gate_vertices(
                            hedge_one
                        )
                    ),
                    second_vertex=min(
                        gain_mgr.distribution.circuit.get_gate_vertices(
                            hedge_two
                        )
                    ),
                    qubit_vertex=qubit_vertex
                )

                # Check that neither of the hyperedges one wishes to merge
                # require H embeddings, nor that there is a H gate between
                # them.
                if not (
                    gain_mgr.distribution.circuit.requires_h_embedded_cu1(
                        hedge_one
                    )
                    or gain_mgr.distribution.circuit.requires_h_embedded_cu1(
                        hedge_two
                    )
                ) and not (
                    OpType.H in [
                        command.op.type for command in intermediate_commands
                    ]
                ):

                    # Merge if doing so reduces the cost of the distribution.
                    if gain_mgr.merge_hyperedge_gain(
                        [hedge_one, hedge_two]
                    ) > 0:
                        new_hyperedge = gain_mgr.merge_hyperedge(
                            [hedge_one, hedge_two])

                        # Remove old and add new hyperedge
                        hedge_list.pop(0)
                        hedge_list.insert(0, new_hyperedge)
