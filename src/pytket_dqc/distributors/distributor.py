from abc import ABC, abstractmethod
from pytket_dqc import DistributedCircuit


class Distributor(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def distribute(self, dist_circ: DistributedCircuit) -> dict:
        pass
