from __future__ import annotations

from pytket_dqc.distributors import Distributor
from pytket.routing import route  # type:ignore
from pytket.passes import (  # type:ignore
    DecomposeSwapsToCXs,
    RebaseQuil,
    PlacementPass,
)
from pytket_dqc.placement import Placement

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pytket_dqc import DistributedCircuit
    from pytket_dqc.networks import NISQNetwork


class Routing(Distributor):
    def __init__(self):
        pass

    def distribute(
        self,
        dist_circ: DistributedCircuit,
        network: NISQNetwork
    ) -> Placement:

        arch, node_qubit_map, pl = network.get_placer()

        routed_circ = dist_circ.circuit.copy()

        PlacementPass(pl).apply(routed_circ)
        routed_circ = route(routed_circ, arch)
        DecomposeSwapsToCXs(arch).apply(routed_circ)
        RebaseQuil().apply(routed_circ)

        print("arch nodes", arch.nodes)
        print("server qubits", network.server_qubits)
        print("node_qubit_map", node_qubit_map)

        node_server_map = {}
        for node in routed_circ.qubits:
            qubit_found_in = [
                server
                for server, qubits
                in network.server_qubits.items()
                if node_qubit_map[node] in qubits
            ]
            assert len(qubit_found_in) == 1
            node_server_map[node] = qubit_found_in[0]

        print("node_server_map", node_server_map)

        dist_circ.reset(routed_circ)
        print("vertex_circuit_map", dist_circ.vertex_circuit_map)

        placement_dict = {}
        for vertex, vertex_info in dist_circ.vertex_circuit_map.items():
            if vertex_info['type'] == 'qubit':
                placement_dict[vertex] = node_server_map[vertex_info['node']]
            elif vertex_info['type'] == 'gate':
                node = vertex_info['command'].qubits[0]
                placement_dict[vertex] = node_server_map[node]
            else:
                raise Exception("Vertex type not recognised")

        print(placement_dict)
        placement = Placement(placement_dict)

        return placement
