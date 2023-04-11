from __future__ import annotations

import networkx as nx  # type: ignore

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pytket_dqc.placement import Placement


class ModuleNetwork:
    """Class for the management of networks of quantum computers.
    """

    def __init__(self, server_coupling: list[list[int]]):
        """Initialisation function.

        :param server_coupling: List of pairs of server indices. Each pair
            specifies that there is a connection between those two servers.
        :type server_coupling: list[list[int]]
        :raises Exception: Raised if internal lists are not a pair.
        :raises Exception: Raised if the network is unconnected.
        """

        for server_list in server_coupling:
            if not len(server_list) == 2:
                raise Exception(
                    "server_coupling should be a list of pairs of servers.")

        self.server_coupling = server_coupling

        # Check that the resulting network is connected.
        # TODO: We may be able to drop this condition.
        if not nx.is_connected(self.get_server_nx()):
            raise Exception("This server network is unconnected.")

    def __eq__(self, other):
        """Check equality based on equality of components"""
        if isinstance(other, ModuleNetwork):
            return self.server_coupling == other.server_coupling
        return False

    def is_placement(self, placement: Placement) -> bool:
        """Checks that placement is valid for this network. In particular
        check that all of the servers used by the placement are indeed in
        this network.

        :param placement: Placement onto network.
        :type placement: Placement
        :return: Is placement valid.
        :rtype: bool
        """

        valid = True

        # Check that every server vertices are placed onto is in this network.
        server_list = self.get_server_list()
        for server in placement.placement.values():
            if server not in server_list:
                valid = False

        return valid

    def get_server_list(self) -> list[int]:
        """Return list of servers.

        :return: List of server indices.
        :rtype: list[int]
        """

        # Servers are saved in server_coupling as edges. Extract vertices here.
        expanded_coupling = [
            server for coupling in self.server_coupling for server in coupling
        ]
        return list(set(expanded_coupling))

    # TODO: This could probably be simplified. I don't know that anyone will
    # use this function. Could just be combined into draw.
    def get_server_nx(self) -> nx.Graph:
        """Return networkx graph of server network.

        :return: networkx graph of server network.
        :rtype: nx.Graph
        """

        G = nx.Graph()
        for edge in self.server_coupling:
            G.add_edge(edge[0], edge[1])
        return G

    def draw_module_network(self) -> None:
        """Draw server network using networkx draw method.
        """

        G = self.get_server_nx()
        nx.draw(G, with_labels=True, pos=nx.nx_agraph.graphviz_layout(G))
