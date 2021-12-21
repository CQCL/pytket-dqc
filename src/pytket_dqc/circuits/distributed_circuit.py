from .hypergraph import Hypergraph
from pytket.predicates import GateSetPredicate  # type: ignore
from pytket import OpType, Circuit

gateset_pred = GateSetPredicate(
    {OpType.Rx, OpType.CZ, OpType.Rz, OpType.Measure}
)


class DistributedCircuit:
    def __init__(self, circuit: Circuit):

        if not gateset_pred.verify(circuit):
            raise Exception("The inputted circuit is not in a valid gateset.")

        self.circuit = circuit
        self.__from_circuit()

    def get_hypergraph(self) -> Hypergraph:
        return self.hypergraph

    def get_circuit(self) -> Circuit:
        return self.circuit

    def __from_circuit(self):

        n_qubits = self.circuit.n_qubits

        command_list_count = []
        CZ_count = 0
        for command in self.circuit.get_commands():
            if command.op.type == OpType.CZ:
                command_list_count.append(
                    {"command": command, "CZ count": CZ_count}
                )
                CZ_count += 1
            else:
                command_list_count.append({"command": command, "CZ count": -1})

        self.hypergraph = Hypergraph()

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
