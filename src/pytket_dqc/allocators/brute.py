from __future__ import annotations

from pytket_dqc.allocators import Allocator
import itertools
from pytket_dqc.placement import Placement

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pytket_dqc import HypergraphCircuit
    from pytket_dqc.networks import NISQNetwork


class Brute(Allocator):
    """Brute force allocator which searches through all placements
    for the best one.
    """

    def __init__(self) -> None:
        pass

    def allocate(
        self,
        dist_circ: HypergraphCircuit,
        network: NISQNetwork,
        **kwargs
    ) -> Placement:
        """Distribute quantum circuit by looking at all possible placements
        and returning the one with the lowest cost.

        :param dist_circ: Circuit to distribute.
        :type dist_circ: HypergraphCircuit
        :param network: Network onto which ``dist_circ`` should be distributed.
        :type network: NISQNetwork
        :raises Exception: Raised if no valid placement could be found.
        :return: Placement of ``dist_circ`` onto ``network``.
        :rtype: Placement
        """

        if not network.can_implement(dist_circ):
            raise Exception(
                "This circuit cannot be implemented on this network."
            )

        # List of all vertices to be placed
        vertex_list = dist_circ.vertex_list
        # List of all servers which could be used
        server_list = network.get_server_list()

        # Initialise list of all valid placements
        valid_placement_list = []

        # TODO: It would be preferable to only check placement which are valid.
        # It may also be more memory efficient to not store a list of all
        # valid placements but to check their cost as they are generated.

        # Iterate over all placements, even those that are not valid.
        # Determin if they are valid, and add them to list if so.
        for placement_set in itertools.product(
            server_list,
            repeat=len(vertex_list)
        ):

            placement_list = list(placement_set)

            # build placement from list of vertices and servers.
            placement_dict = {vertex: server for vertex, server in zip(
                vertex_list, placement_list)}
            placement = Placement(placement_dict)

            # Append to list is placement is valid.
            if placement.is_valid(dist_circ, network):
                valid_placement_list.append(placement)

        # Raise exception if there are no valid placements. This could happen
        # if the network is too small for example.
        if len(valid_placement_list) == 0:
            raise Exception("No valid placement could be found.")

        # Initialise minimum placement cost to be that of the first
        # valid placement.
        minimum_placement_cost = valid_placement_list[0].cost(
            dist_circ, network)
        minimum_cost_placement = valid_placement_list[0]
        # Check if any of the other valid placements have smaller cost.
        for placement in valid_placement_list[1:]:
            placement_cost = placement.cost(dist_circ, network)
            if placement_cost < minimum_placement_cost:
                minimum_cost_placement = placement
                minimum_placement_cost = placement_cost
            if placement_cost == 0:
                break

        assert minimum_cost_placement.is_valid(dist_circ, network)
        return minimum_cost_placement
