from __future__ import annotations

from abc import ABC, abstractmethod

from typing import TYPE_CHECKING, Any
if TYPE_CHECKING:
    from pytket_dqc.placement import Placement


class Distributor(ABC):
    def __init__(self):
        pass

    # TODO: Correct type here to be any subclass of ServerNetwork
    @abstractmethod
    def distribute(
        self,
        dist_circ: Any,
        network: Any
    ) -> Placement:
        pass
