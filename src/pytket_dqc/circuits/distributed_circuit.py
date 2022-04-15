from __future__ import annotations

from .hypergraph import Hypergraph
from pytket.predicates import GateSetPredicate  # type: ignore
from pytket import OpType, Circuit
from pytket.circuit import CustomGateDef  # type: ignore
from scipy.stats import unitary_group  # type: ignore
import numpy as np
from pytket.passes import DecomposeBoxes, auto_rebase_pass  # type: ignore
import networkx as nx  # type: ignore
import random
from pytket_dqc.utils import dqc_gateset_predicate, _cost_from_circuit

from typing import TYPE_CHECKING, cast
if TYPE_CHECKING:
    from pytket.circuit import Command, QubitRegister  # type: ignore
    from pytket import Qubit  # type: ignore
    from pytket_dqc.placement import Placement

def_circ = Circuit(2)
def_circ.add_barrier([0, 1])

start_proc = CustomGateDef.define("StartingProcess", def_circ, [])
end_proc = CustomGateDef.define("EndingProcess", def_circ, [])
telep_proc = CustomGateDef.define("Teleportation", def_circ, [])

def_circ = Circuit(2)
def_circ.add_barrier([0, 1])

start_proc = CustomGateDef.define("StartingProcess", def_circ, [])
end_proc = CustomGateDef.define("EndingProcess", def_circ, [])
telep_proc = CustomGateDef.define("Teleportation", def_circ, [])


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

    def __str__(self):
        out_string = super().__str__()
        out_string += f"\nVertex Circuit Map: {self.vertex_circuit_map}"
        out_string += "\nCircuit: " + self.circuit.__str__()
        return out_string

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
        self.commands: list[dict[str,
                                 Union[Command, str, int, list[Qubit]]]] = []
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

    def get_qubit_vertices(self):
        return [
            vertex
            for vertex in self.vertex_list
            if self.vertex_circuit_map[vertex]['type'] == 'qubit'
        ]

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

    def is_qubit_vertex(self, vertex: int) -> bool:
        """Checks if the given vertex corresponds to a qubit.

        :param vertex: Vertex to be checked.
        :type vertex: int
        :return: Is it a qubit vertex.
        :rtype: bool
        """
        return self.vertex_circuit_map[vertex]['type'] == 'qubit'

    def from_circuit(self):
        """Method to create a hypergraph from a circuit.

        :raises Exception: Raised if the circuit whose hypergraph is to be
        created is not in the valid gate set.
        """

        if not dqc_gateset_predicate.verify(self.circuit):
            raise Exception("The inputted circuit is not in a valid gateset.")

        two_q_gate_count = 0
        # For each command in the circuit, add the command to a list.
        # If the command is a CZ, CRz or CX, store n, where the
        # command is the nth 2 qubit gate in the circuit.
        for command in self.circuit.get_commands():
            if command.op.type in [OpType.CZ, OpType.CRz, OpType.CX]:
                self.commands.append(
                    {"command": command, "two q gate count": two_q_gate_count}
                )
                two_q_gate_count += 1
            else:
                self.commands.append({"command": command})

        # Construct the hypergraph corresponding to this circuit.
        # For each qubit, add commands acting on the qubit in an uninterrupted
        # sequence (i.e. not separated by single qubit gates) to the
        # same hyperedge, along with the qubit. If the gate is a CX,
        # add the gate vertex when the control is intercepted, and add a new
        # weight 2 hyper edge if the control is intercepted. The weight 2
        # hyperedge corresponds to performing a possible teleportation.
        for qubit_index, qubit in enumerate(self.circuit.qubits):

            self.add_qubit_vertex(qubit_index, qubit)

            hyperedge = [qubit_index]
            # Gather all of the commands acting on qubit.
            qubit_commands = [
                {"command_index": command_index, "command": command}
                for command_index, command in enumerate(self.commands)
                if qubit in command["command"].qubits
            ]

            for command_dict in qubit_commands:
                command = command_dict['command']
                command_index = command_dict['command_index']
                # If the command is a CZ gate add it to the current working
                # hyperedge.
                if command["command"].op.type in [OpType.CZ, OpType.CRz]:
                    vertex = command["two q gate count"] + \
                        self.circuit.n_qubits
                    self.add_gate_vertex(vertex, command['command'])
                    hyperedge.append(vertex)
                    self.commands[command_index]['vertex'] = vertex
                    self.commands[command_index]['type'] = 'distributed gate'
                # If the command is a CX, add it to the current
                # working hyperedge, if the working qubit is the control.
                # Otherwise start a fresh weight 2 hyper edge, add the two
                # vertex hyperedge consisting of the gate and the qubit, and
                # start a fresh hyper edge again.
                # TODO: Note that this method of adding a CX is very
                # lazy. Indeed, in the case where a teleportation is required,
                # a new hyper edge need not be started, as other gates which
                # follow may also benefit from the teleportation.
                elif command["command"].op.type in [
                    OpType.CX
                ]:
                    # Check if working qubit is the control
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
                    self.commands[command_index]['vertex'] = vertex
                    self.commands[command_index]['type'] = 'distributed gate'
                # Otherwise (which is to say if a single qubit gate is
                # encountered) add the hyperedge to the hypergraph and start
                # a new working hyperedge.
                else:
                    if len(hyperedge) > 1:
                        self.add_hyperedge(hyperedge)
                        hyperedge = [qubit_index]
                    self.commands[command_index]['type'] = '1q local gate'

            # If there is an hyperedge that has not been added once all
            # commands have bee iterated through, add it now.
            if len(hyperedge) > 1:
                self.add_hyperedge(hyperedge)

    def to_relabeled_registers(self, placement: Placement):

        if not self.is_placement(placement):
            raise Exception("This is not a valid placement for this circuit.")

        qubit_vertices = self.get_qubit_vertices()
        servers_used = set([
            server
            for vertex, server in placement.placement.items()
            if vertex in qubit_vertices
        ])
        server_to_vertex_dict = {
            server: [
                vertex
                for vertex in qubit_vertices
                if placement.placement[vertex] == server
            ]
            for server in servers_used
        }

        circ = Circuit()
        server_register_map = {}
        for server, vertex_list in server_to_vertex_dict.items():
            server_register_map[server] = circ.add_q_register(
                f'Server {server}', len(vertex_list))

        qubit_qubit_map = {}
        for server, register in server_register_map.items():
            for i, qubit_vertex in enumerate(server_to_vertex_dict[server]):
                qubit_qubit_map[self.vertex_circuit_map[qubit_vertex]
                                ['node']] = register[i]

        for gate in self.circuit.get_commands():
            circ.add_gate(gate.op, [qubit_qubit_map[orig_qubit]
                          for orig_qubit in gate.args])

        return circ

    def to_pytket_circuit(self, placement: Placement):

        # Initial check that placement is valid
        if not self.is_placement(placement):
            raise Exception("This is not a valid placement for this circuit.")

        # A dictionary mapping servers to the qubit vertices it contains
        server_to_qubit_vertex_list = {
            server: [
                vertex
                for vertex in self.get_qubit_vertices()
                if placement.placement[vertex] == server
            ]
            for server in set(placement.placement.values())
        }

        # New circuit including distribution gates
        circ = Circuit()

        # A dictionary mapping servers to the qubit registers they contain.
        server_to_register = {}

        # A dictionary mapping (server, hyperedge) pairs to link qubits.
        # The first index is the server the link qubit is contained in.
        # The second index is the hyperedge the link qubit is consumed by.
        server_to_link_register: dict[int, QubitRegister] = {}

        # Here a register is defined for each server. It contains a number of
        # qubits equal to the number of qubit vertices which are placed in the
        # server, +1 qubit for each hyperedge which requires a qubit be
        # moved to this server. Note that this means there is an additional
        # link qubit per e-bit. By the completion of this loop the circuit
        # has all the qubits necessary to complete the computaitons.
        for server, qubit_vertex_list in server_to_qubit_vertex_list.items():

            # Add a register for all of the qubits assigned to this server.
            server_to_register[server] = circ.add_q_register(
                f'Server {server}', len(qubit_vertex_list))

            server_to_link_register[server] = {}
            # For each hyperedge, add the necessary link qubits
            for index, hyperedge in enumerate(self.hyperedge_list):

                # List of gate vertices in this hyperedge
                gate_vertex_list = [
                    vertex
                    for vertex in cast(list[int], hyperedge['hyperedge'])
                    if self.vertex_circuit_map[vertex]['type'] == 'gate'
                ]

                # Find the one unique vertex in the hyperedge which
                # corresponds to a qubit.
                hyperedge_qubit_vertex_list = [
                    vertex
                    for vertex in cast(list[int], hyperedge['hyperedge'])
                    if vertex not in gate_vertex_list
                ]
                assert len(hyperedge_qubit_vertex_list) == 1
                hyperedge_qubit_vertex = hyperedge_qubit_vertex_list[0]

                # Add a link qubits if the qubit of the hyperedge is not
                # placed in this server, but there is a gate vertex in this
                # hyperedge which is placed in this server.
                if not (placement.placement[hyperedge_qubit_vertex] == server):

                    unique_server_used = set([
                        placement.placement[gate_vertex]
                        for gate_vertex in gate_vertex_list
                    ])
                    if server in unique_server_used:
                        register = circ.add_q_register(
                            f'Server {server} Link Edge {index}', 1)
                        server_to_link_register[server][index] = register[0]

        # Dictionary mapping qubit vertices in hypergraph to server qubit
        # registers
        qubit_vertex_to_server_qubit = {}
        for server, register in server_to_register.items():
            for i, vertex in enumerate(server_to_qubit_vertex_list[server]):
                qubit_vertex_to_server_qubit[vertex] = register[i]

        # Dictionary mapping qubits in original circuit to
        # qubits in new registers
        circuit_qubit_to_server_qubit = {
            self.vertex_circuit_map[qubit_vertex]['node']: qubit
            for qubit_vertex, qubit in qubit_vertex_to_server_qubit.items()
        }

        # Dictionary mapping gate vertices to command information. Later new
        # arguments are added to this to show then new qubits on which the
        # command acts.
        gate_vertex_to_command = {
            vertex: data
            for vertex, data in self.vertex_circuit_map.items()
            if data['type'] == 'gate'}

        # Dictionary mapping circuit qubits to hypergraph vertices
        circuit_qubit_to_vertex = {
            info['node']: vertex
            for vertex, info in self.vertex_circuit_map.items()
            if info['type'] == 'qubit'
        }

        # This loop updates, for each command, the qubits on which the
        # commands act.
        for gate_vertex, command in gate_vertex_to_command.items():

            # Get original circuit qubits used by command
            orig_circuit_qubit = command['command'].qubits

            # Get servers to which qubits belongs
            orig_server = [placement.placement[circuit_qubit_to_vertex[qubit]]
                           for qubit in orig_circuit_qubit]

            # The server in which the gate is acted
            gate_server = placement.placement[gate_vertex]

            new_qubit = []
            for server, qubit in zip(orig_server, orig_circuit_qubit):
                # If the original circuit qubit has been placed on
                # the same server that the gate has been assigned to, then
                # use the relabeled server qubit as the control
                if server == gate_server:
                    new_qubit.append(circuit_qubit_to_server_qubit[qubit])
                # If not, move the control to the link qubits of the server
                # to which the gate has been assigned
                else:
                    qubit_vertex = circuit_qubit_to_vertex[qubit]
                    # Find the hyperedge to which the qubit and gate vertex
                    # belong. This should be unique. Use this to locate
                    # the correct link qubit.
                    for index, hyperedge in enumerate(self.hyperedge_list):
                        if (
                            (
                                gate_vertex
                                in cast(list[int], hyperedge['hyperedge'])
                            )
                            and
                            (
                                qubit_vertex
                                in cast(list[int], hyperedge['hyperedge'])
                            )
                        ):
                            new_qubit.append(
                                server_to_link_register[gate_server][index])

            assert len(orig_server) == len(new_qubit)

            # Update the dictionary from gate vertices to commands
            gate_vertex_to_command[gate_vertex]['args'] = new_qubit

        # A list of commands defining the new circuit. Add original commands
        # to the list of new commands
        new_command_list = [{
            "vertex": gate_vertex,
            "op": command['command'].op,
            'args':command['args'],
            'role':'gate'}
            for gate_vertex, command in gate_vertex_to_command.items()]

        # For each hyperedge add the necessary distributed operations.
        for edge_index, edge in enumerate(self.hyperedge_list):

            # List of the subset of vertices which correspond to gates.
            gate_vertex_list = [
                vertex
                for vertex in cast(list[int], edge['hyperedge'])
                if self.vertex_circuit_map[vertex]['type'] == 'gate'
            ]

            # Find the qubit vertex in the hyperedge.
            qubit_vertex_set = set(
                cast(list[int], edge['hyperedge'])) - set(gate_vertex_list)
            assert len(qubit_vertex_set) == 1
            qubit_vertex_list = list(qubit_vertex_set)
            qubit_vertex = qubit_vertex_list[0]

            qubit_server = placement.placement[qubit_vertex]

            # List of servers used by gates in hyperedge
            unique_server_list = list(
                set(
                    [
                        placement.placement[vertex]
                        for vertex in gate_vertex_list
                    ]
                )
            )
            if qubit_server in unique_server_list:
                unique_server_list.remove(qubit_server)

            first_found = False
            first = 0

            # Look through the list of commands to find the first gate in
            # the hyperedge
            while not first_found:
                if new_command_list[first]['role'] == 'gate':
                    if new_command_list[first]['vertex'] in gate_vertex_list:
                        first_found = True
                    else:
                        first += 1
                else:
                    first += 1

            # For every server used by gates in the hyperedge, add a starting
            # process or teletortation before all of the gates are acted.
            for server in unique_server_list:
                if edge['weight'] == 1:
                    new_command = {
                        'role': 'start',
                        'args': [
                            qubit_vertex_to_server_qubit[qubit_vertex],
                            server_to_link_register[server][edge_index]
                        ]
                    }
                    new_command_list.insert(first, new_command)

                elif edge['weight'] == 2:
                    new_command = {
                        'role': 'teleport',
                        'args': [
                            qubit_vertex_to_server_qubit[qubit_vertex],
                            server_to_link_register[server][edge_index]
                        ]
                    }
                    new_command_list.insert(first, new_command)
                else:
                    raise Exception(
                        "The operation for this weight is not known")

            last_found = False
            last = len(new_command_list) - 1

            # Look through the list of commands to find the last gate in
            # the hyperedge
            while not last_found:
                if new_command_list[last]['role'] == 'gate':
                    if new_command_list[last]['vertex'] in gate_vertex_list:
                        last_found = True
                    else:
                        last -= 1
                else:
                    last -= 1

            # For every server used by gates in the hyperedge, add an ending
            # process or teletortation after all of the gates are acted.
            for server in unique_server_list:
                if edge['weight'] == 1:
                    new_command = {
                        'role': 'end',
                        'args': [
                            server_to_link_register[server][edge_index],
                            qubit_vertex_to_server_qubit[qubit_vertex]
                        ]
                    }
                    new_command_list.insert(last+1, new_command)
                elif edge['weight'] == 2:
                    new_command = {
                        'role': 'teleport',
                        'args': [
                            server_to_link_register[server][edge_index],
                            qubit_vertex_to_server_qubit[qubit_vertex]
                        ]
                    }
                    new_command_list.insert(last+1, new_command)

        # For each command in the new command list, add it to the circuit.
        for command in new_command_list:

            # TODO: We can use case statements in python 3.10 I think.
            # We should upgrade at some point.
            if command['role'] == 'gate':
                circ.add_gate(command['op'], command['args'])
            elif command['role'] == 'start':
                circ.add_custom_gate(start_proc, [], command['args'])
            elif command['role'] == 'end':
                circ.add_custom_gate(end_proc, [], command['args'])
            elif command['role'] == 'teleport':
                circ.add_custom_gate(telep_proc, [], command['args'])
            else:
                raise Exception('This role has not been defined')

        return circ


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
