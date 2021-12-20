from pytket_dqc.hypergraph import Hypergraph
from pytket.predicates import GateSetPredicate
from pytket import OpType, Circuit

gateset_pred = GateSetPredicate(
    {OpType.Rx, OpType.CZ, OpType.Rz, OpType.Measure}
)

class DistributedCircuit:

    def __init__(self):

        self.circuit = Circuit()
        self.hypergraph = Hypergraph()

    def get_hypergraph(self):
        return self.hypergraph

    def from_circuit(self, circuit: Circuit):

        self.circuit = circuit

        if not gateset_pred.verify(circuit):
            raise Exception("The inputted circuit is not in a valid gateset.")

        n_qubits = self.circuit.n_qubits

        command_list_count = []
        CZ_count = 0
        for command in self.circuit.get_commands():
            if command.op.type == OpType.CZ:
                command_list_count.append({"command": command, "CZ count": CZ_count})
                CZ_count += 1
            else:
                command_list_count.append({"command": command, "CZ count": -1})

        for qubit_index, qubit in enumerate(self.circuit.qubits):

            self.hypergraph.add_vertex(qubit_index)

            hyperedge = [qubit_index]
            qubit_commands = [
                command
                for command in command_list_count
                if qubit in command["command"].qubits
            ]

            for command in qubit_commands:
                if command["command"].op.type == OpType.CZ:
                    vertex = command["CZ count"] + n_qubits
                    self.hypergraph.add_vertex(vertex)
                    hyperedge.append(vertex)
                elif len(hyperedge) > 1:
                    self.hypergraph.add_hyperedge(hyperedge)
                    hyperedge = [qubit_index]

            if len(hyperedge) > 1:
                self.hypergraph.add_hyperedge(hyperedge)