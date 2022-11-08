from .refiner import Refiner
from pytket_dqc.circuits.distribution import Distribution
from pytket_dqc.allocators import GainManager


class MergeDType(Refiner):

    def refine(self, initial_distribution: Distribution):

        assert initial_distribution.circuit._sorted_hedges_predicate()

        gain_mgr = GainManager(initial_distribution=initial_distribution)
            
        for qubit_vertex in initial_distribution.circuit.get_qubit_vertices():

            hedge_list = [
                hedge
                for hedge in initial_distribution.circuit.hyperedge_list
                if initial_distribution.circuit.get_qubit_vertex(hedge) == qubit_vertex
            ]

            while len(hedge_list) >= 2:
                
                hedge_one = hedge_list.pop(0)
                hedge_two = hedge_list[0]

                if not (
                    initial_distribution.circuit.requires_h_embedded_cu1(
                        hedge_one
                    ) 
                    and initial_distribution.circuit.requires_h_embedded_cu1(
                        hedge_two
                    )
                ):

                    if gain_mgr.merge_hyperedge_gain([hedge_one, hedge_two]) > 0:
                        new_hyperedge = gain_mgr.merge_hyperedge([hedge_one, hedge_two])
                        hedge_list.pop(0)
                        hedge_list.insert(0, new_hyperedge)

        assert initial_distribution.circuit._sorted_hedges_predicate()
