from __future__ import annotations

from pytket_dqc.refiners import Refiner, BoundaryReallocation

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pytket_dqc import Distribution


class DetachedGates(Refiner):
    """An alias for ``BoundaryReallocation`` with ``reallocate_qubits``
    set to False. This refiner can optimise gate distribution so that detached
    gates may be used, even in the presence of embedding.
    """

    def refine(self, distribution: Distribution, **kwargs) -> bool:
        """An alias for boundary reallocation with no movements of
        qubit vertices.

        :param distribution: Distribution to refine.
        :type distribution: Distribution

        :key num_rounds: Max number of refinement rounds. Default is 1000.
        :key stop_parameter: Real number in [0,1]. If proportion of moves
            in a round is smaller than this number, do no more rounds. Default
            is 0.05.
        :key seed: Seed for randomness. Default is None.
        :key cache_limit: The maximum size of the set of servers whose cost is
            stored in cache; see GainManager. Default value is 5.

        :return: Distribution where the placement updated.
        :rtype: Distribution
        """

        num_rounds = kwargs.get("num_rounds", 1000)
        stop_parameter = kwargs.get("stop_parameter", 0.05)
        seed = kwargs.get("seed", None)
        cache_limit = kwargs.get("cache_limit", None)

        return BoundaryReallocation().refine(
            distribution,
            reallocate_qubits=False,
            num_rounds=num_rounds,
            stop_parameter=stop_parameter,
            seed=seed,
            cache_limit=cache_limit,
        )
