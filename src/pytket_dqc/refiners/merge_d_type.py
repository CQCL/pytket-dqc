from .refiner import Refiner
from pytket_dqc.circuits.distribution import Distribution


class MergeDTypeRefiner(Refiner):

    def refine(self, distribution: Distribution):

        for qubit in distribution.circuit.circuit.qubits:
            print(qubit)
