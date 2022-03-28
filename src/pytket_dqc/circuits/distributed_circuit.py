from __future__ import annotations

from .hypergraph import Hypergraph
from pytket.predicates import GateSetPredicate  # type: ignore
from pytket import OpType, Circuit
from scipy.stats import unitary_group  # type: ignore
import numpy as np
from pytket.passes import DecomposeBoxes  # type: ignore
from pytket.circuit import Unitary2qBox  # type: ignore
import networkx as nx  # type: ignore
import random
from pytket.passes import auto_rebase_pass

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pytket.circuit import Command  # type: ignore
    from pytket import Qubit  # type: ignore
    from pytket_dqc.placement import Placement

allowed_gateset = {OpType.Rx, OpType.CZ,
                   OpType.Rz, OpType.Measure, OpType.QControlBox}
gateset_pred = GateSetPredicate(allowed_gateset)


class DistributedCircuit(Hypergraph):
    """Class representing circuit to be distributed on a network.
    DistributedCircuit is a child of Hypergraph. DistributedCircuit adds
    additional information on top of Hypergraph which describes the
    correspondence to a circuit.

    :param circuit: Circuit to be distributed.
    :type circuit: Circuit
    :param vertex_circuit_map: Map from hypergraph vertices to circuit
        commands.
    :type vertex_circuit_map: dict[int, dict]
    """

    def __init__(self, circuit: Circuit):
        """ Initialisation function

        :param circuit: Circuit to be distributed.
        :type circuit: Circuit
        """

        self.reset(circuit)

    def place(self, placement: Placement):

        if not self.is_placement(placement):
            raise Exception("This is not a valid placement of this circuit")

    def reset(self, circuit: Circuit):
        """Reset object with a new circuit.

        :param circuit: Circuit to reinitialise object with.
        :type circuit: Circuit
        """

        super().__init__()

        self.circuit = circuit
        self.vertex_circuit_map: dict[int, dict] = {}
        self.from_circuit()

    def add_qubit_vertex(self, vertex: int, qubit: Qubit):
        """Add a vertex to the underlying hypergraph which corresponds to a
        qubit. Adding vertices in this way allow for the distinction between
        qubit vertices and gate vertices to be tracked.

        :param vertex: Vertex to be added to hypergraph.
        :type vertex: int
        :param qubit: Qubit to which the vertex should correspond
        :type qubit: Qubit
        """
        self.add_vertex(vertex)
        self.vertex_circuit_map[vertex] = {'type': 'qubit', 'node': qubit}

    def add_gate_vertex(self, vertex: int, command: Command):
        """Add a vertex to the underlying hypergraph which corresponds to a
        gate. Adding vertices in this way allows for the distinction between
        qubit and gate vertices to be tracked.

        :param vertex: Vertex to be added to the hypergraph.
        :type vertex: int
        :param command: Command to which the vertex corresponds.
        :type command: Command
        """
        self.add_vertex(vertex)
        self.vertex_circuit_map[vertex] = {'type': 'gate', 'command': command}

    def from_circuit(self):
        """Method to create a hypergraph from a circuit.

        :raises Exception: Raised if the circuit whose hypergraph is to be
        created is not in the Rx, Rz, CZ, QControlBox gate set.
        :raises Exception: Raised if any of the QControlBox gates do not have
        a single control and a single target.
        """

        if not gateset_pred.verify(self.circuit):
            raise Exception("The inputted circuit is not in a valid gateset.")

        command_list_count = []
        two_q_gate_count = 0
        # For each command in the circuit, add the command to a list.
        # If the command is a CZ gate or QControlBox, store n, where the
        # command is the nth 2 qubit gate in the circuit.
        for command in self.circuit.get_commands():
            if command.op.type == OpType.CZ:
                command_list_count.append(
                    {"command": command, "two q gate count": two_q_gate_count}
                )
                two_q_gate_count += 1
            elif command.op.type == OpType.QControlBox:
                if len(command.qubits) != 2:
                    raise Exception(
                        "QControlBox must have one target and one control")
                command_list_count.append(
                    {"command": command, "two q gate count": two_q_gate_count}
                )
                two_q_gate_count += 1
            else:
                command_list_count.append({"command": command})

        # Construct the hypergraph corresponding to this circuit.
        # For each qubit, add commands acting on the qubit in an uninterrupted
        # sequence (i.e. not separated by single qubit gates) to the
        # same hyperedge, along with the qubit. If the gate is a QControlBox,
        # add the gate vertex when the control is intercepted, and add a new
        # weight 2 hyper edge if the control is intercepted. The weight 2
        # hyperedge corresponds to performing a possible teleportation.
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
                    vertex = command["two q gate count"] + \
                        self.circuit.n_qubits
                    self.add_gate_vertex(vertex, command['command'])
                    hyperedge.append(vertex)
                # If the command is a QControlBox, add it to the current
                # working hyperedge, is the working qubit is the control.
                # Otherwise start a fresh weight 2 hyper edge, add the two
                # vertex hyperedge consisting of the gate and the qubit, and
                # start a fresh hyper edge again.
                # TODO: Note that this method of adding a QControlBox is very
                # lazy. Indeed, in the case where a teleportation is required,
                # a new hyper edge need not be started, as other gates which
                # follow may also benefit from the teleportation.
                elif command["command"].op.type == OpType.QControlBox:
                    if qubit == command['command'].qubits[0]:
                        vertex = command["two q gate count"] + \
                            self.circuit.n_qubits
                        self.add_gate_vertex(vertex, command['command'])
                        hyperedge.append(vertex)
                    else:
                        # Add current working hyperedge.
                        if len(hyperedge) > 1:
                            self.add_hyperedge(hyperedge)
                        # Add two vertex weight 2 hyperedge
                        vertex = command["two q gate count"] + \
                            self.circuit.n_qubits
                        self.add_gate_vertex(vertex, command['command'])
                        hyperedge = [qubit_index, vertex]
                        self.add_hyperedge(hyperedge, weight=2)
                        # Start a fresh hyperedge
                        hyperedge = [qubit_index]
                # Otherwise add the hyperedge to the hypergraph and start
                # a new working hyperedge.
                elif len(hyperedge) > 1:
                    self.add_hyperedge(hyperedge)
                    hyperedge = [qubit_index]

            if len(hyperedge) > 1:
                self.add_hyperedge(hyperedge)


class RandomDistributedCircuit(DistributedCircuit):
    """Generates circuit to be distributed, where the circuit is a random
    circuit of the form used in quantum volume experiments.
    """

    def __init__(self, n_qubits: int, n_layers: int):
        """Initialisation function.

        :param n_qubits: The number of qubits the circuit covers.
        :type n_qubits: int
        :param n_layers: The number of layers of random 2-qubit gates.
        :type n_layers: int
        """

        circ = Circuit(n_qubits)

        for _ in range(n_layers):

            # Generate a random bipartition (a collection of pairs) of
            # the qubits.
            qubits = np.random.permutation([i for i in range(n_qubits)])
            qubit_pairs = [[qubits[i], qubits[i + 1]]
                           for i in range(0, n_qubits - 1, 2)]

            # Act a random 2-qubit unitary between each pair.
            for pair in qubit_pairs:

                SU4 = unitary_group.rvs(4)  # random unitary in SU4
                SU4 = SU4 / (np.linalg.det(SU4) ** 0.25)

                circ.add_unitary2qbox(Unitary2qBox(SU4), *pair)

        # Rebase to a valid gate set.
        DecomposeBoxes().apply(circ)
        auto_rebase_pass({OpType.CZ, OpType.Rz, OpType.Rx}).apply(circ)

        super().__init__(circ)


class CyclicDistributedCircuit(DistributedCircuit):
    """Particular instance of the DistributedCircuit class, where the circuit
    is constructed from CZ gates acting in a loop accross all qubits.
    """

    def __init__(self, n_qubits: int, n_layers: int):
        """Initialisation function.

        :param n_qubits: The number of qubits the circuit covers.
        :type n_qubits: int
        :param n_layers: The number of layers of loops of CZ gates.
        :type n_layers: int
        """

        circ = Circuit(n_qubits)
        for _ in range(n_layers):
            for qubit in range(n_qubits-1):
                circ.CZ(qubit, qubit+1)
            circ.CZ(n_qubits-1, 0)

        super().__init__(circ)


class RegularGraphDistributedCircuit(DistributedCircuit):
    """DistributedCircuit constructed by acting CZ gates between qubits which
    neighbour each other in a random regular graph.
    """

    def __init__(
        self,
        n_qubits: int,
        degree: int,
        n_layers: int,
        seed: int = None,
    ):
        """Initialisation function

        :param n_qubits: The number of qubits on which the circuit acts.
        :type n_qubits: int
        :param degree: The degree of the random regular graph.
        :type degree: int
        :param n_layers: The number of random regular graphs to generate.
        :type n_layers: int
        :param seed: Seed for the random generation of regular graphs,
            defaults to None
        :type seed: int, optional
        """

        circ = Circuit(n_qubits)
        for _ in range(n_layers):
            G = nx.generators.random_graphs.random_regular_graph(
                degree, n_qubits, seed=seed)
            for v, u in G.edges():
                circ.CZ(v, u)
                circ.Rx(random.uniform(0, 2), random.choice(list(G.nodes)))
                circ.Rz(random.uniform(0, 2), random.choice(list(G.nodes)))

        super().__init__(circ)
