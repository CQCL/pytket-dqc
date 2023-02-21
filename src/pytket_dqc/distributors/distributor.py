from __future__ import annotations

from abc import ABC, abstractmethod

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pytket_dqc.circuits.distribution import Distribution
    from pytket_dqc.networks import NISQNetwork
    from pytket import Circuit


class Distributor(ABC):
    """Abstract class defining the structure of distributors. Distributors
    are complete recommended DQC workflows.
    """

    def __init__(self) -> None:
        pass

    @abstractmethod
    def distribute(
        self, circ: Circuit, network: NISQNetwork, **kwargs
    ) -> Distribution:
        pass
