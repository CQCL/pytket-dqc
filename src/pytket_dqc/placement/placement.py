from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pytket_dqc.networks import NISQNetwork
    from pytket_dqc.circuits import DistributedCircuit


class Placement:

    def __init__(self, placement: dict[int, int]):
        self.placement = placement

    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(other, Placement):
            return self.placement == other.placement
        return False

    def valid(self, circuit: DistributedCircuit, network: NISQNetwork) -> bool:

        if not circuit.is_placement(self):
            return False
        elif not network.is_placement(self):
            return False
        else:
            is_valid = True

        # Check that no more qubits are allotted to a server than can be
        # accommodated.
        for server in list(set(self.placement.values())):
            vertices = [vertex for vertex in self.placement.keys()
                        if self.placement[vertex] == server]
            qubits = [
                vertex
                for vertex in vertices
                if circuit.vertex_circuit_map[vertex]['type'] == 'qubit'
            ]
            if len(qubits) > len(network.server_qubits[server]):
                is_valid = False

        return is_valid
