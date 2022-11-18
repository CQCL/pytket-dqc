from .refiner import Refiner
from pytket_dqc.circuits.distribution import Distribution


class SequenceRefiner(Refiner):
    """Performs sequence of Refiners as provided by the user.
    """

    def __init__(self, refiner_list: list[Refiner]):
        """SequenceRefiner is initialised with a list of Refiners to be
        run in sequence.

        :param refiner_list: List of Refiners to be run in sequence.
        :type refiner_list: list[Refiner]
        """

        self.refiner_list = refiner_list

    def refine(self, distribution: Distribution) -> bool:
        """Perform each of the refinements in the provided sequence.

        :param distribution: Distribution to be refined.
        :type distribution: Distribution
        :return: True if if any of the Refiners
            in the sequence makes a refinement. False otherwise.
        :rtype: bool
        """

        refinement_made = False
        for refiner in self.refiner_list:
            refinement_made |= refiner.refine(distribution)

        return refinement_made
