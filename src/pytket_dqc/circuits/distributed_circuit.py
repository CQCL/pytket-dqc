from __future__ import annotations

from .hypergraph import Hypergraph
from pytket.predicates import GateSetPredicate  # type: ignore
from pytket import OpType

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pytket.circuit import Command  # type: ignore
    from pytket import Circuit, Qubit  # type:ignore

gateset_pred = GateSetPredicate(
    {OpType.Rx, OpType.CZ, OpType.Rz, OpType.Measure}
)


class DistributedCircuit(Hypergraph):
    def __init__(self, circuit: Circuit):

        self.reset(circuit)

    def reset(self, circuit: Circuit):

        super().__init__()

        self.circuit = circuit
        self.vertex_circuit_map: dict[int, dict] = {}
        self.from_circuit()

    def add_qubit_vertex(self, vertex: int, qubit: Qubit):
        self.add_vertex(vertex)
        self.vertex_circuit_map[vertex] = {'type': 'qubit', 'node': qubit}

    def add_gate_vertex(self, vertex: int, command: Command):
        self.add_vertex(vertex)
        self.vertex_circuit_map[vertex] = {'type': 'gate', 'command': command}

    def from_circuit(self):

        if not gateset_pred.verify(self.circuit):
            raise Exception("The inputted circuit is not in a valid gateset.")

        command_list_count = []
        CZ_count = 0
        # For each command in the circuit, add the command to a list.
        # If the command is a CZ gate, store n, where the command is
        # the nth CZ gate in the circuit.
        for command in self.circuit.get_commands():
            if command.op.type == OpType.CZ:
                command_list_count.append(
                    {"command": command, "CZ count": CZ_count}
                )
                CZ_count += 1
            else:
                command_list_count.append({"command": command, "CZ count": -1})

        # Construct the hypergraph corresponding to this circuit.
        # For each qubit, add commands acting on the qubit in an uninterrupted
        # sequence (i.e. not separated by single qubit gates) to the
        # same hyperedge, along with the qubit.
        for qubit_index, qubit in enumerate(self.circuit.qubits):

            self.add_qubit_vertex(qubit_index, qubit)

            hyperedge = [qubit_index]
            # Gather all of the commands acting on qubit.
            qubit_commands = [
                command
                for command in command_list_count
                if qubit in command["command"].qubits
            ]

            for command in qubit_commands:
                # If the command is a CZ gate add it to the current working
                # hyperedge.
                if command["command"].op.type == OpType.CZ:
                    vertex = command["CZ count"] + self.circuit.n_qubits
                    self.add_gate_vertex(vertex, command['command'])
                    hyperedge.append(vertex)
                # Otherwise add the hyperedge to the hypergraph and start
                # a new working hyperedge.
                elif len(hyperedge) > 1:
                    self.add_hyperedge(hyperedge)
                    hyperedge = [qubit_index]

            if len(hyperedge) > 1:
                self.add_hyperedge(hyperedge)
