from __future__ import annotations

from pytket_dqc.distributors import Distributor
from pytket.passes import (  # type:ignore
    DecomposeSwapsToCXs,
    PlacementPass,
    RoutingPass
)
from pytket.passes import auto_rebase_pass
from pytket_dqc.placement import Placement

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pytket_dqc import DistributedCircuit
    from pytket_dqc.networks import NISQNetwork
from pytket.circuit import OpType  # type:ignore


class Routing(Distributor):
    """Distribute quantum circuits using routing tools available in
    `tket <https://cqcl.github.io/tket/pytket/api/routing.html>`_. Note that
    this distributor will alter the initial circuit.
    """
    def __init__(self) -> None:
        pass

    def distribute(
        self,
        dist_circ: DistributedCircuit,
        network: NISQNetwork,
        **kwargs
    ) -> Placement:
        """Distribute quantum circuits using routing tools available in
        `tket <https://cqcl.github.io/tket/pytket/api/routing.html>`_. Note
        that this distributor will alter the initial circuit.

        :param dist_circ: Circuit to distribute.
        :type dist_circ: DistributedCircuit
        :param network: Network onto which ``dist_circ`` should be distributed.
        :type network: NISQNetwork
        :return: Placement of ``dist_circ`` onto ``network``.
        :rtype: Placement
        """

        arch, node_qubit_map, pl = network.get_placer()

        routed_circ = dist_circ.circuit.copy()

        # Place and route circuit onto architecture.
        PlacementPass(pl).apply(routed_circ)

        RoutingPass(arch).apply(routed_circ)

        DecomposeSwapsToCXs(arch).apply(routed_circ)
        # TODO: Add some optimisation to account for impact of adding SWAPs.
        auto_rebase_pass({OpType.CZ, OpType.Rz, OpType.Rx}).apply(routed_circ)

        # Map of vertices to servers
        node_server_map = {}
        # for each qubit, find the server which it has been placed in.
        for node in routed_circ.qubits:
            # List of servers where vertex has been found.
            qubit_found_in = [
                server
                for server, qubits
                in network.server_qubits.items()
                if node_qubit_map[node] in qubits
            ]
            assert len(qubit_found_in) == 1
            node_server_map[node] = qubit_found_in[0]

        dist_circ.reset(routed_circ)

        placement_dict = {}
        # For each vertex in the circuit hypergraph, place it in a server.
        for vertex, vertex_info in dist_circ.vertex_circuit_map.items():
            # If the vertex is a qubit use node_server_map
            if vertex_info['type'] == 'qubit':
                placement_dict[vertex] = node_server_map[vertex_info['node']]
            # If the vertex is a gate, use the same server as it's
            # control qubits.
            elif vertex_info['type'] == 'gate':
                node = vertex_info['command'].qubits[0]
                placement_dict[vertex] = node_server_map[node]
            else:
                raise Exception("Vertex type not recognised")

        placement = Placement(placement_dict)
        assert placement.is_valid(dist_circ, network)
        return placement
