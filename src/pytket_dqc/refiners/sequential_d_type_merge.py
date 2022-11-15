from .refiner import Refiner
from pytket_dqc.circuits.distribution import Distribution
from pytket_dqc.allocators import GainManager
from pytket import OpType


class SequentialDTypeMerge(Refiner):
    """Refiner merging sequentially neighbouring packets
    """

    def refine(self, distribution: Distribution):
        """Merges neighbouring packets when no Hadamard act between them
        Packets concerning a particular qubit are
        all merged where possible.

        :param distribution: Distribution whose packets should be
            merged.
        :type distribution: Distribution
        """

        gain_mgr = GainManager(initial_distribution=distribution)

        # Iterate through all qubits, merging packets.
        for qubit_vertex in gain_mgr.distribution.circuit.get_qubit_vertices():

            # List of hyperedges acting on qubit
            hedge_list = gain_mgr.distribution.circuit.hyperedge_dict[
                qubit_vertex
            ].copy()

            # iterates through list of hyperedges. If the first element in
            # list can be merged with the next, do so, add the merged
            # hyperedge to the start of the list, and remove the two original
            # hyperedges. If it cannot be merged with the next, remove it
            # or the second element from the list. The second is removed in
            # the case where the first hops over the second.
            # Repear until the list is empty.
            while len(hedge_list) >= 2:

                # Hyperedges to try to merge.
                hedge_one = hedge_list[0]
                hedge_two = hedge_list[1]

                hedge_one_gates = gain_mgr.distribution.circuit.get_gate_vertices(hedge_one)  # noqa: E501
                hedge_two_gates = gain_mgr.distribution.circuit.get_gate_vertices(hedge_two)  # noqa: E501

                intermediate_commands = gain_mgr.distribution.circuit.get_intermediate_commands(  # noqa: E501
                    first_vertex=max(
                        vertex for vertex in hedge_one_gates
                        if vertex < min(hedge_two_gates)
                    ),
                    second_vertex=min(hedge_two_gates),
                    qubit_vertex=qubit_vertex
                )

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

                # Remove whichever of the hyperedges has the gate occurring
                # latest in the circuit.
                elif max(hedge_one_gates) < max(hedge_two_gates):
                    hedge_list.pop(0)
                else:
                    hedge_list.pop(1)

        assert gain_mgr.distribution.is_valid()
