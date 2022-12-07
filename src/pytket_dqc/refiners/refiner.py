from abc import ABC, abstractmethod
from pytket_dqc.circuits.distribution import Distribution


class Refiner(ABC):
    """Abstract base class defining the behaviors of Refiners, which perform
    in place processing on a `Distribution`.
    """

    @abstractmethod
    def refine(self, distribution: Distribution, **kwargs) -> None:
        pass
