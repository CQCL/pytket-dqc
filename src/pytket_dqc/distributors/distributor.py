from __future__ import annotations

from abc import ABC, abstractmethod

from typing import TYPE_CHECKING, Any
if TYPE_CHECKING:
    from pytket_dqc.placement import Placement


class Distributor(ABC):
    """Abstract class defining the structure of distributors which distribute
    quantum circuits on networks.
    """
    def __init__(self):
        pass

    # TODO: Correct type here to be any subclass of ServerNetwork
    @abstractmethod
    def distribute(
        self,
        dist_circ: Any,
        network: Any,
        **kwargs
    ) -> Placement:
        """Method returning placement of circuit onto network.

        :param dist_circ: Circuit to distribute.
        :type dist_circ: Any
        :param network: Network onto which ``dist_circ`` should be distributed.
        :type network: Any
        :return: Placement of ``dist_circ`` onto ``network``.
        :rtype: Placement
        """
        pass
