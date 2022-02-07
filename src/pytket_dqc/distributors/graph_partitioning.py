from __future__ import annotations

import kahypar as kahypar  # type:ignore
from pytket_dqc.distributors import Distributor
from pytket_dqc.placement import Placement

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pytket_dqc import DistributedCircuit
    from pytket_dqc.networks import ServerNetwork


class GraphPartitioning(Distributor):
    """Distribution technique, making use of existing tools for hypergraph
    partitioning available through the `kahypar <https://kahypar.org/>`_
    package. This distributor is not guaranteed to return a valid placement
    as it will perform load balancing, which is to say an even placement of
    vertices onto servers.
    """

    def __init__(self, epsilon: float = 0.03) -> None:
        """Initialisation function.

        :param epsilon: Load imbalance tolerance, defaults to 0.03
        :type epsilon: float, optional
        """
        self.epsilon = epsilon

    # TODO: dist_circ does not need to be a DistributedCircuit and could be a
    # Hypergraph. Is there a way of specifying this in the typing?
    def distribute(
        self,
        dist_circ: DistributedCircuit,
        network: ServerNetwork,
        **kwargs
    ) -> Placement:
        """Distribute ``dist_circ`` onto ``network`` using graph partitioning
        tools available in `kahypar <https://kahypar.org/>`_ package. This
        may not return a valid placement.

        :param dist_circ: Circuit to distribute.
        :type dist_circ: DistributedCircuit
        :param network: Network onto which ``dist_circ`` should be placed.
        :type network: ServerNetwork
        :return: Placement of ``dist_circ`` onto ``network``.
        :rtype: Placement
        """

        if not dist_circ.is_valid():
            raise Exception("This hypergraph is not valid.")

        hyperedge_indices, hyperedges = dist_circ.kahypar_hyperedges()

        num_hyperedges = len(hyperedge_indices) - 1
        num_vertices = len(list(set(hyperedges)))
        num_servers = len(network.get_server_list())

        hypergraph = kahypar.Hypergraph(
            num_vertices,
            num_hyperedges,
            hyperedge_indices,
            hyperedges,
            num_servers
        )

        context = kahypar.Context()
        context.loadINIconfiguration("km1_kKaHyPar_sea20.ini")
        context.setK(num_servers)
        context.setEpsilon(self.epsilon)
        context.suppressOutput(True)

        kahypar.partition(hypergraph, context)

        partition_list = [hypergraph.blockID(i)
                          for i in range(hypergraph.numNodes())]

        placement_dict = {i: server for i, server in enumerate(partition_list)}

        return Placement(placement_dict)
