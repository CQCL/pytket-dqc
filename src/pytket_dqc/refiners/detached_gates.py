from __future__ import annotations

from pytket_dqc.refiners import Refiner, BoundaryReallocation

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pytket_dqc import Distribution


class DetachedGates(Refiner):
    """An alias for ``BoundaryReallocation`` with ``fixed_vertices`` set to
    the list of qubit-vertices and embedded gate-vertices. This refiner can
    optimise gate distribution so that detached gates may be used.
    """

    def refine(self, distribution: Distribution, **kwargs) -> bool:
        """An alias for boundary reallocation with no movements of
        qubit vertices or embedded gates.
        Key arguments (kwargs) are passed directly to `BoundaryReallocation`;
        the arguments it accepts are described in its documentation.

        :param distribution: Distribution to refine.
        :type distribution: Distribution

        :return: Distribution where the placement updated.
        :rtype: Distribution
        """

        fixed_vertices = (
            distribution.circuit.get_qubit_vertices()
            + distribution.circuit.get_all_h_embedded_gate_vertices()
        )

        return BoundaryReallocation().refine(
            distribution,
            fixed_vertices=fixed_vertices,
            **kwargs
        )
