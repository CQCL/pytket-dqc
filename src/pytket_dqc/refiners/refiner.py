from abc import ABC, abstractmethod
from pytket_dqc.circuits.distribution import Distribution


class Refiner(ABC):
    """Abstract base class defining the behaviors of Refiners, which perform
    in place processing on a :class:`.Distribution`.
    """

    @abstractmethod
    def refine(self, distribution: Distribution, **kwargs) -> bool:
        """Perform in place refinement of a :class:`.Distribution`.

        :param distribution: Distribution to be refined.
        :type distribution: Distribution
        :return: True if a refinement has been performed. False otherwise.
        :rtype: bool
        """
        pass
