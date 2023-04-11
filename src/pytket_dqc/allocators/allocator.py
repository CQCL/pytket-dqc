from __future__ import annotations

from abc import ABC, abstractmethod

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pytket_dqc.circuits.distribution import Distribution
    from pytket_dqc.networks import HeterogeneousNetwork
    from pytket import Circuit


class Allocator(ABC):
    """Abstract class defining the structure of allocators which allocate
    quantum circuits on networks.
    """

    def __init__(self) -> None:
        pass

    # TODO: Correct type here to be any subclass of ServerNetwork
    @abstractmethod
    def allocate(
        self, circ: Circuit, network: HeterogeneousNetwork, **kwargs
    ) -> Distribution:
        pass
