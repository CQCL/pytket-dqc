from .refiner import Refiner
from pytket_dqc.circuits.distribution import Distribution

class RepeatRefiner(Refiner):

    def __init__(self, refiner: Refiner):

        self.refiner = refiner

    def refine(self, distribution: Distribution):

        refinement_made = self.refiner.refine(distribution)
        while refinement_made:
            refinement_made = self.refiner.refine(distribution)

        return self.refiner.refine(distribution)