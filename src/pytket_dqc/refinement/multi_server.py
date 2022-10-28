from .refinement import Refinement
from pytket_dqc.circuits.distribution import Distribution


class MultiServerRefinement(Refinement):

    def refine(self, distribution: Distribution):

        for command in distribution.circuit._circuit.get_commands():
            print("command", command)
