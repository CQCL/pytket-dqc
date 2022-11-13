from .refiner import Refiner
from pytket_dqc.circuits.distribution import Distribution
from pytket_dqc.allocators import GainManager
from pytket import OpType


class SequentialMerge(Refiner):
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
            # or the second element from the list. The second is removed in
            # the case where the first hops over the second.
            # Repear until the list is empty.
            while len(hedge_list) >= 2:

                # Hyperedges to try to merge.
                hedge_one = hedge_list[0]
                hedge_two = hedge_list[1]

                hedge_one_gates = gain_mgr.distribution.circuit.get_gate_vertices(hedge_one)  # noqa: E501
                hedge_two_gates = gain_mgr.distribution.circuit.get_gate_vertices(hedge_two)  # noqa: E501

                # commands between hyperedges.
                intermediate_commands = gain_mgr.distribution.circuit.get_intermediate_commands(  # noqa: E501
                    first_vertex=max(hedge_one_gates),
                    second_vertex=min(hedge_two_gates),
                    qubit_vertex=qubit_vertex
                )

                # Determine if the hyperedges follow each other. This is to
                # say that it's checked all the gate vertices of one
                # of the hyperedges must follow all the gate vertices
                # of the other. If not it is assumed that hedge_one hops
                # over hedge_two
                are_consecutive = all([
                    all([
                        vertex_one < vertex_two
                        for vertex_one in hedge_one_gates
                    ]) for vertex_two in hedge_two_gates
                ]) or all([
                    all([
                        vertex_two < vertex_one
                        for vertex_one in hedge_one_gates
                    ]) for vertex_two in hedge_two_gates
                ])

                # Check that there is no H gate between
                # the hyperedges, and that they are consecutive.
                if not (
                    OpType.H in [
                        command.op.type for command in intermediate_commands
                    ]
                ) and are_consecutive:

                    # Merge if doing so does not increase
                    # the cost of the distribution.
                    if gain_mgr.merge_hyperedge_gain(
                        [hedge_one, hedge_two]
                    ) >= 0:

                        new_hyperedge = gain_mgr.merge_hyperedge(
                            [hedge_one, hedge_two]
                        )

                        # Remove old and add new hyperedges
                        hedge_list.pop(0)
                        hedge_list.pop(0)
                        hedge_list.insert(0, new_hyperedge)

                # If they are not consecutive it is assumed that hedge_one hops
                # over hedge_two. If that is the case hedge_two should be
                # removed to allow for continued sequential merging.
                elif not are_consecutive:
                    hedge_list.pop(1)
                # else remove hedge_one.
                else:
                    hedge_list.pop(0)
