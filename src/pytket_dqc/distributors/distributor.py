from abc import ABC, abstractmethod
from pytket_dqc import DistributedCircuit
from pytket_dqc.networks import NISQNetwork
from typing import Union


class Distributor(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def distribute(
        self,
        dist_circ: DistributedCircuit,
        network: Union[NISQNetwork]
    ) -> dict[int, int]:
        pass
