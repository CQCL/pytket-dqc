from .refiners import Refiner
from pytket_dqc.circuits.distribution import Distribution


class MultiServerRefinement(Refiner):

    def refine(self, distribution: Distribution):

        for command in distribution.circuit._circuit.get_commands():
            print("command", command)
