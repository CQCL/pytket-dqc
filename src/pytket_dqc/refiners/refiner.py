from abc import ABC, abstractmethod
from pytket_dqc.circuits.distribution import Distribution


class Refiner(ABC):
    """Abstract base class defining the behaviors of Refiners, which perform
    processing on a `Distribution`.
    """

    def __init__(self) -> None:
        pass

    @abstractmethod
    def refine(self, distribution: Distribution, **kwargs) -> None:
        pass
