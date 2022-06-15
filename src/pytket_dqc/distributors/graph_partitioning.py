from __future__ import annotations

import kahypar as kahypar  # type:ignore
from pytket_dqc.distributors import Distributor
from .ordered import Ordered
from pytket_dqc.placement import Placement
import importlib_resources


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pytket_dqc import DistributedCircuit
    from pytket_dqc.networks import NISQNetwork


class GraphPartitioning(Distributor):
    """Distribution technique, making use of existing tools for hypergraph
    partitioning available through the `kahypar <https://kahypar.org/>`_
    package. This distributor will ignore weighted hypergraphs and assume
    all hyperedges have weight 1. This distributor will ignore the
    connectivity of the NISQNetwork.
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
        self, dist_circ: DistributedCircuit, network: NISQNetwork, **kwargs
    ) -> Placement:
        """Distribute ``dist_circ`` onto ``network`` using graph partitioning
        tools available in `kahypar <https://kahypar.org/>`_ package. This
        may not return a valid placement.

        :param dist_circ: Circuit to distribute.
        :type dist_circ: DistributedCircuit
        :param network: Network onto which ``dist_circ`` should be placed.
        :type network: ServerNetwork

        :key ini_path: Path to kahypar ini file.

        :return: Placement of ``dist_circ`` onto ``network``.
        :rtype: Placement

        :key seed: Seed for randomness. Default is None
        """

        if not dist_circ.is_valid():
            raise Exception("This hypergraph is not valid.")

        seed = kwargs.get("seed", None)

        if not (len(dist_circ.hyperedge_list) == 0):

            hyperedge_indices, hyperedges = dist_circ.kahypar_hyperedges()

            num_hyperedges = len(hyperedge_indices) - 1
            num_vertices = len(list(set(hyperedges)))
            server_list = network.get_server_list()
            num_servers = len(server_list)
            server_sizes = [len(network.server_qubits[s]) for s in server_list]
            # For now, all hyperedges are assumed to have the same weight
            hyperedge_weights = [1 for i in range(0, num_hyperedges)]
            # Qubit vertices are given weight 1, gate vertices are given
            # weight 0
            num_qubits = len(dist_circ.circuit.qubits)
            vertex_weights = [1 for i in range(0, num_qubits)] + [
                0 for i in range(num_qubits, num_vertices)
            ]
            # TODO: the weight assignment to vertices assumes that the index
            # of the qubit vertices range from 0 to `num_qubits`, and the
            # rest of them correspond to gates. This is currently guaranteed
            # by construction i.e. method `from_circuit()`; we might want
            # to make this more robust.

            hypergraph = kahypar.Hypergraph(
                num_vertices,
                num_hyperedges,
                hyperedge_indices,
                hyperedges,
                num_servers,
                hyperedge_weights,
                vertex_weights,
            )

            context = kahypar.Context()

            package_path = importlib_resources.files("pytket_dqc")
            default_ini = f"{package_path}/distributors/km1_kKaHyPar_sea20.ini"
            ini_path = kwargs.get("ini_path", default_ini)
            context.loadINIconfiguration(ini_path)

            context.setK(num_servers)
            context.setEpsilon(self.epsilon)
            context.setCustomTargetBlockWeights(server_sizes)
            context.suppressOutput(True)
            if seed is not None:
                context.setSeed(seed)

            kahypar.partition(hypergraph, context)

            partition_list = [
                hypergraph.blockID(i) for i in range(hypergraph.numNodes())
            ]

            placement_dict = {i: server for i,
                              server in enumerate(partition_list)}
            placement = Placement(placement_dict)

        else:

            placement = Ordered().distribute(dist_circ, network)

        return placement
