from .refiner import Refiner
from pytket_dqc.circuits.distribution import Distribution
from pytket_dqc.allocators import GainManager
from pytket import OpType


class NeighbouringDTypeMerge(Refiner):
    """Refiner merging neighbouring packets when no Hadamard act between them.
    For each qubit in the circuit, and given the packets ordered as they
    appear in the corresponding hypergraph, this refiner will attempt
    to merge packets appearing consecutively in that ordering.
    """

    def refine(self, distribution: Distribution) -> bool:
        """Merges neighbouring packets when no Hadamard acts between them.

        :param distribution: Distribution whose packets should be merged.
        :type distribution: Distribution
        :return: True if a refinement has been performed. False otherwise.
        :rtype: bool
        """

        gain_mgr = GainManager(initial_distribution=distribution)
        refinement_made = False

        # Iterate through all qubits, merging packets.
        for qubit_vertex in gain_mgr.distribution.circuit.get_qubit_vertices():

            # List of hyperedges acting on qubit
            hedge_list = gain_mgr.distribution.circuit.hyperedge_dict[
                qubit_vertex
            ].copy()

            # Iterates through list of hyperedges. If the first element in
            # list can be merged with the next, do so, add the merged
            # hyperedge to the start of the list, and remove the two original
            # hyperedges. If it cannot be merged with the next, remove it
            # or the second element from the list. The second is removed in
            # the case where the first hops over the second.
            # Repeat until the list is empty.
            while len(hedge_list) >= 2:

                # Hyperedges to try to merge.
                hedge_one = hedge_list[0]
                hedge_two = hedge_list[1]

                hedge_one_gates = gain_mgr.distribution.circuit.get_gate_vertices(hedge_one)  # noqa: E501
                hedge_two_gates = gain_mgr.distribution.circuit.get_gate_vertices(hedge_two)  # noqa: E501

                # Gates intermediate between the fist gate in hedge_two,
                # and the last gate in hedge_one which predeeds it.
                intermediate_commands = gain_mgr.distribution.circuit.get_intermediate_commands(  # noqa: E501
                    first_vertex=max(
                        vertex for vertex in hedge_one_gates
                        if vertex < min(hedge_two_gates)
                    ),
                    second_vertex=min(hedge_two_gates),
                    qubit_vertex=qubit_vertex
                )

                # If there are no Hadamard gates preventing the merge, and
                # it it beneficial to do so, merge the hyperedges.
                if OpType.H not in [
                    command.op.type for command in intermediate_commands
                ] and gain_mgr.merge_hyperedge_gain(
                    [hedge_one, hedge_two]
                ) >= 0:

                    new_hyperedge = gain_mgr.merge_hyperedge(
                        [hedge_one, hedge_two]
                    )

                    # Remove old and add new hyperedges
                    del hedge_list[:2]
                    hedge_list.insert(0, new_hyperedge)
                    refinement_made = True

                # Keep whichever of the hyperedges has the gate occurring
                # latest in the circuit.
                elif max(hedge_one_gates) < max(hedge_two_gates):
                    hedge_list.pop(0)
                else:
                    hedge_list.pop(1)

        assert gain_mgr.distribution.is_valid()
        return refinement_made
