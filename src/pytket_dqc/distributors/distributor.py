from __future__ import annotations

from abc import ABC, abstractmethod
from pytket_dqc import DistributedCircuit
from pytket_dqc.networks import NISQNetwork
from typing import Union

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pytket_dqc.placement import Placement


class Distributor(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def distribute(
        self,
        dist_circ: DistributedCircuit,
        network: Union[NISQNetwork]
    ) -> Placement:
        pass
