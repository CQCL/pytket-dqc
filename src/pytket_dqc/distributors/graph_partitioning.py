from __future__ import annotations

import kahypar as kahypar  # type:ignore
from pytket_dqc.distributors import Distributor
from pytket_dqc.placement import Placement

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pytket_dqc import DistributedCircuit
    from pytket_dqc.networks import ServerNetwork


class GraphPartitioning(Distributor):

    def __init__(self, epsilon: float = 0.03):
        self.epsilon = epsilon

    # TODO: dist_circ does not need to be a DistributedCircuit and could be a
    # Hypergraph. Is there a way of specifying this in the typing?
    def distribute(
        self,
        dist_circ: DistributedCircuit,
        network: ServerNetwork,
        **kwargs
    ) -> Placement:

        if not dist_circ.is_valid():
            raise Exception("This hypergraph is not valid.")

        hyperedge_indices, hyperedges = dist_circ.kahypar_hyperedges()

        num_nets = len(hyperedge_indices) - 1
        num_nodes = len(list(set(hyperedges)))

        k = len(network.get_server_list())

        hypergraph = kahypar.Hypergraph(
            num_nodes, num_nets, hyperedge_indices, hyperedges, k)

        context = kahypar.Context()
        context.loadINIconfiguration("km1_kKaHyPar_sea20.ini")

        context.setK(k)
        context.setEpsilon(self.epsilon)
        context.suppressOutput(True)

        kahypar.partition(hypergraph, context)

        partition_list = [hypergraph.blockID(i)
                          for i in range(hypergraph.numNodes())]

        placement_dict = {i: server for i, server in enumerate(partition_list)}

        return Placement(placement_dict)
