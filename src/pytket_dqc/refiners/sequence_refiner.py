from .refiner import Refiner
from pytket_dqc.circuits.distribution import Distribution


class SequenceRefiner(Refiner):

    def __init__(self, refiner_list: list[Refiner]):

        self.refiner_list = refiner_list

    def refine(self, distribution: Distribution):

        refinement_made = False
        for refiner in self.refiner_list:
            refinement_made |= refiner.refine(distribution)

        return refinement_made
