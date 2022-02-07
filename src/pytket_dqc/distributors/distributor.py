from __future__ import annotations

from abc import ABC, abstractmethod

from typing import TYPE_CHECKING, Any
if TYPE_CHECKING:
    from pytket_dqc.placement import Placement


class Distributor(ABC):
    """Abstract class defining the structure of distributors which distribute
    quantum circuits on networks.
    """
    def __init__(self) -> None:
        pass

    # TODO: Correct type here to be any subclass of ServerNetwork
    @abstractmethod
    def distribute(
        self,
        dist_circ: Any,
        network: Any,
        **kwargs
    ) -> Placement:
        pass
