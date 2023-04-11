from __future__ import annotations

import random
import kahypar as kahypar  # type:ignore
from pytket_dqc.allocators import Allocator, GainManager
from pytket_dqc.placement import Placement
from pytket_dqc.circuits import HypergraphCircuit, Distribution
import importlib_resources
from pytket import Circuit


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pytket_dqc.networks import HeterogeneousNetwork


class HypergraphPartitioning(Allocator):
    """Distribution technique, making use of existing tools for hypergraph
    partitioning available through the `KaHyPar <https://kahypar.org/>`_
    package. This allocator will ignore weights on hyperedges and
    assume all hyperedges have weight 1. This allocator will ignore the
    connectivity of the HeterogeneousNetwork.
    """

    def allocate(
        self, circ: Circuit, network: HeterogeneousNetwork, **kwargs
    ) -> Distribution:
        """Distribute ``circ`` onto ``network``. The distribution
        is found by KaHyPar using the connectivity metric. All-to-all
        connectivity of the network of modules is assumed; you may wish
        to run the refiner ``BoundaryReallocation`` on the output
        ``Distribution`` to take the network topology into account.

        :param circ: Circuit to distribute.
        :type circ: pytket.Circuit
        :param network: Network onto which ``circ`` should be placed.
        :type network: HeterogeneousNetwork

        :key ini_path: Path to kahypar ini file. Default points to the
            ini file within the pytket-dqc repository.
        :key seed: Seed for randomness. Default is None

        :return: Distribution of ``circ`` onto ``network``.
        :rtype: Distribution
        """

        dist_circ = HypergraphCircuit(circ)
        if not network.can_implement(dist_circ):
            raise Exception(
                "This circuit cannot be implemented on this network."
            )

        package_path = importlib_resources.files("pytket_dqc")
        default_ini = f"{package_path}/allocators/km1_kKaHyPar_sea20.ini"
        ini_path = kwargs.get("ini_path", default_ini)
        seed = kwargs.get("seed", None)

        # First step is to call KaHyPar using the connectivity metric (i.e. no
        # knowledge about network topology other than server sizes)
        placement = self.initial_distribute(
            dist_circ, network, ini_path, seed=seed
        )

        distribution = Distribution(dist_circ, placement, network)
        self.make_valid(distribution, seed=seed)

        return distribution

    def make_valid(self, distribution: Distribution, **kwargs):
        """Since KaHyPar does not guarantee that the requirement on server
        capacity will be satisfied, we enforce this ourselves.
        However, it usually does satisfy the requirement and the following
        code often does nothing or it moves very few vertices.
        """
        seed = kwargs.get("seed", None)
        if seed is not None:
            random.seed(seed)

        # We will use a ``GainManager`` to manage the calculation of gains
        # (and management of pre-computed values) in a transparent way
        gain_manager = GainManager(distribution)

        for vertex in gain_manager.qubit_vertices:
            # If moving in place is not valid then the current server is full
            if not gain_manager.is_move_valid(
                vertex, gain_manager.current_server(vertex)
            ):
                # Then, find the first server where the move would be valid
                for server in gain_manager.occupancy.keys():
                    if gain_manager.is_move_valid(vertex, server):
                        # Move ``vertex`` to a server with free spaces and
                        # return control to the outer loop
                        gain_manager.move_vertex(vertex, server)
                        break
        # Notice that the moves have been arbitrary, i.e. we have not
        # calculated gains. This is fine since the vertices we moved will
        # likely be boundary vertices; the following rounds of the
        # refinement algorithm will move them around to optimise gains.
        # At the end of the previous subroutine, no server should be
        # overpopulated.
        assert gain_manager.distribution.is_valid()
        # GainManager has updated ``distribution`` in place:
        assert gain_manager.distribution is distribution

    def initial_distribute(
        self,
        dist_circ: HypergraphCircuit,
        network: HeterogeneousNetwork,
        ini_path: str,
        **kwargs,
    ) -> Placement:
        """Distribute ``dist_circ`` onto ``network`` using graph partitioning
        tools available in `kahypar <https://kahypar.org/>`_ package. The
        placement returned is not taking into account network topology.
        However, it does take into account server sizes.

        :param dist_circ: Circuit to distribute.
        :type dist_circ: HypergraphCircuit
        :param network: Network onto which ``dist_circ`` should be placed.
        :type network: HeterogeneousNetwork
        :param ini_path: Path to kahypar ini file.
        :type ini_path: str

        :key seed: Seed for randomness. Default is None

        :return: Placement of ``dist_circ`` onto ``network``.
        :rtype: Placement
        """

        if not dist_circ.is_valid():
            raise Exception("This hypergraph is not valid.")

        seed = kwargs.get("seed", None)

        # This should only arise if the circuit is completely empty.
        if len(dist_circ.hyperedge_list) == 0:
            assert dist_circ._circuit == Circuit()
            return Placement(dict())
        else:
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
            num_qubits = len(dist_circ.get_qubit_vertices())
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
            default_ini = f"{package_path}/allocators/km1_kKaHyPar_sea20.ini"
            ini_path = kwargs.get("ini_path", default_ini)
            context.loadINIconfiguration(ini_path)

            context.setK(num_servers)
            context.setCustomTargetBlockWeights(server_sizes)
            context.suppressOutput(True)
            if seed is not None:
                context.setSeed(seed)

            kahypar.partition(hypergraph, context)

            partition_list = [
                hypergraph.blockID(i) for i in range(hypergraph.numNodes())
            ]

            placement_dict = {
                i: server for i, server in enumerate(partition_list)
            }
            placement = Placement(placement_dict)

        return placement
