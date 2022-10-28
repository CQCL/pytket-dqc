from abc import ABC, abstractmethod
from pytket_dqc.circuits.distribution import Distribution


class Refinement(ABC):

    def __init__(self) -> None:
        pass

    @abstractmethod
    def refine(self, distribution: Distribution) -> None:
        pass
