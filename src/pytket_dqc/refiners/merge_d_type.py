from .refiner import Refiner
from pytket_dqc.circuits.distribution import Distribution
from pytket_dqc.allocators import GainManager
from pytket import OpType


class MergeDType(Refiner):

    def refine(self, initial_distribution: Distribution):

        init_circ = initial_distribution.circuit
        gain_mgr = GainManager(initial_distribution=initial_distribution)

        for qubit_vertex in init_circ.get_qubit_vertices():

            hedge_list = [
                hedge
                for hedge in init_circ.hyperedge_list
                if init_circ.get_qubit_vertex(hedge) == qubit_vertex
            ]

            while len(hedge_list) >= 2:

                hedge_one = hedge_list.pop(0)
                hedge_two = hedge_list[0]

                intermediate_commands = init_circ.get_intermediate_commands(
                    first_vertex=max(init_circ.get_gate_vertices(hedge_one)),
                    second_vertex=min(init_circ.get_gate_vertices(hedge_two)),
                    qubit_vertex=qubit_vertex
                )

                if not (
                    init_circ.requires_h_embedded_cu1(hedge_one)
                    or init_circ.requires_h_embedded_cu1(hedge_two)
                ) and not (
                    OpType.H in [
                        command.op.type for command in intermediate_commands
                    ]
                ):

                    if gain_mgr.merge_hyperedge_gain(
                        [hedge_one, hedge_two]
                    ) > 0:
                        new_hyperedge = gain_mgr.merge_hyperedge(
                            [hedge_one, hedge_two])
                        hedge_list.pop(0)
                        hedge_list.insert(0, new_hyperedge)
