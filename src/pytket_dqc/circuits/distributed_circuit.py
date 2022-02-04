from __future__ import annotations

from .hypergraph import Hypergraph
from pytket.predicates import GateSetPredicate  # type: ignore
from pytket import OpType, Circuit
from scipy.stats import unitary_group  # type: ignore
import numpy as np
from pytket.passes import DecomposeBoxes, RebaseQuil  # type: ignore
from pytket.circuit import Unitary2qBox  # type: ignore
import networkx as nx  # type: ignore
import random

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pytket.circuit import Command  # type: ignore
    from pytket import Qubit  # type: ignore

allowed_gateset = {OpType.Rx, OpType.CZ, OpType.Rz, OpType.Measure}
gateset_pred = GateSetPredicate(allowed_gateset)


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


class RandomDistributedCircuit(DistributedCircuit):

    def __init__(self, n_qubits, n_layers):

        circ = Circuit(n_qubits)

        for _ in range(n_layers):

            qubits = np.random.permutation([i for i in range(n_qubits)])
            qubit_pairs = [[qubits[i], qubits[i + 1]]
                           for i in range(0, n_qubits - 1, 2)]

            for pair in qubit_pairs:

                SU4 = unitary_group.rvs(4)  # random unitary in SU4
                SU4 = SU4 / (np.linalg.det(SU4) ** 0.25)

                circ.add_unitary2qbox(Unitary2qBox(SU4), *pair)

        DecomposeBoxes().apply(circ)
        RebaseQuil().apply(circ)

        super().__init__(circ)


class CyclicDistributedCircuit(DistributedCircuit):

    def __init__(self, n_qubits, n_layers):

        circ = Circuit(n_qubits)
        for _ in range(n_layers):
            for qubit in range(n_qubits-1):
                circ.CZ(qubit, qubit+1)
            circ.CZ(n_qubits-1, 0)

        super().__init__(circ)


class RegularGraphDistributedCircuit(DistributedCircuit):

    def __init__(self, n_qubits: int, degree: int, n_layers: int, seed=None):

        circ = Circuit(n_qubits)
        for _ in range(n_layers):
            G = nx.generators.random_graphs.random_regular_graph(
                degree, n_qubits, seed=seed)
            for v, u in G.edges():
                circ.CZ(v, u)
                circ.Rx(random.uniform(0, 2), random.choice(list(G.nodes)))
                circ.Rz(random.uniform(0, 2), random.choice(list(G.nodes)))

        super().__init__(circ)
