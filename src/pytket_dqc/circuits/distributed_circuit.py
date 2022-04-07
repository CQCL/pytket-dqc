from __future__ import annotations

from .hypergraph import Hypergraph
from pytket import OpType, Circuit
from pytket.circuit import CustomGateDef
from scipy.stats import unitary_group  # type: ignore
import numpy as np
from pytket.passes import DecomposeBoxes  # type: ignore
from pytket.circuit import Unitary2qBox  # type: ignore
import networkx as nx  # type: ignore
import random
from pytket.passes import auto_rebase_pass
from pytket_dqc.utils.gateset import dqc_gateset_predicate

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pytket.circuit import Command  # type: ignore
    from pytket import Qubit  # type: ignore
    from pytket_dqc.placement import Placement

allowed_gateset = {OpType.Rx, OpType.CZ,
                   OpType.Rz, OpType.Measure, OpType.QControlBox}
gateset_pred = GateSetPredicate(allowed_gateset)

def_circ = Circuit(2)
def_circ.add_barrier([0, 1])

start_proc = CustomGateDef.define("StartingProcess", def_circ, [])
end_proc = CustomGateDef.define("EndingingProcess", def_circ, [])

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

    def from_circuit(self):
        """Method to create a hypergraph from a circuit.

        :raises Exception: Raised if the circuit whose hypergraph is to be
        created is not in the Rx, Rz, CZ, QControlBox gate set.
        :raises Exception: Raised if any of the QControlBox gates do not have
        a single control and a single target.
        """

        if not dqc_gateset_predicate.verify(self.circuit):
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
            elif command.op.type in [OpType.QControlBox, OpType.CX]:
                if len(command.qubits) != 2:
                    raise Exception(
                        "QControlBox must have one target and one control")
                command_list_count.append(
                    {"command": command, "two q gate count": two_q_gate_count}
                )
                two_q_gate_count += 1
            elif command.op.n_qubits >= 2:
                # This elif should never be reached if the gate set predicate
                # has been verified. This is a fail safe.
                raise Exception(
                    "A greater than two qubit command cannot be distributed \
                    if it is not in the valid gate set.")
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
                elif command["command"].op.type in [
                    OpType.QControlBox,
                    OpType.CX
                ]:
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

    def to_pytket_circuit(self, placement):
            
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

        circ = Circuit()
            
        # A dictionary mapping servers to the qubit registers
        # they contain. 
        server_to_register = {}
        server_to_link_register = {}
        # Here a register is defined for each server. It contains a number of qubits
        # equal to the number of qubit vertices which are placed in the server, +1 
        # qubit for each server in the network.
        # TODO: It actually adds a qubit for each server with index smaller than the 
        # server with the highest index which is used. This is not ver nice and should
        # be replaced. Possibly by adding separate registers for the link qubits.
        for server, qubit_vertex_list in server_to_qubit_vertex_list.items():
            # server_to_register[server] = circ.add_q_register(f'Server {server}', len(qubit_vertex_list) + max(placement.placement.values()) + 1)
            server_to_register[server] = circ.add_q_register(f'Server {server}', len(qubit_vertex_list))
            server_to_link_register[server] = {}
            for link_server in set(placement.placement.values()):
                register = circ.add_q_register(f'Server {server} Link {link_server}', 1)
                server_to_link_register[server][link_server] = register[0]

        # Dictionary mapping qubits in original circuit to 
        # qubits in new registers
        circuit_qubit_to_server_qubit = {}
        for server, register in server_to_register.items():
            for i, qubit_vertex in enumerate(server_to_qubit_vertex_list[server]):
                circuit_qubit_to_server_qubit[self.vertex_circuit_map[qubit_vertex]
                                ['node']] = register[i]

        # Dictionary mapping gate vertices to command information
        gate_vertex_to_command = {vertex:data for vertex, data in self.vertex_circuit_map.items() if data['type'] == 'gate'}
        
        # Dictionary mapping circuit qubits to hypergraph vertices
        circuit_qubit_to_vertex = {info['node']:vertex for vertex, info in self.vertex_circuit_map.items() if info['type'] == 'qubit'}
        
        print("server_to_qubit_vertex_list", server_to_qubit_vertex_list)
        print("server_to_register", server_to_register)
        print("server_to_link_register", server_to_link_register)
        print("circuit_qubit_to_server_qubit", circuit_qubit_to_server_qubit)
        print("gate_vertex_to_command", gate_vertex_to_command)
        print("circuit_qubit_to_vertex", circuit_qubit_to_vertex)
        
        for gate_vertex, command in gate_vertex_to_command.items():
            
            # Get circuit qubits used by command
            # TODO: We are assuming 2 qubit gates here which is not ideal
            orig_ctrl_circuit_qubit = command['command'].qubits[0]
            orig_targ_circuit_qubit = command['command'].qubits[1]
            
            # Get servers to which qubit belongs
            orig_ctrl_server = placement.placement[circuit_qubit_to_vertex[orig_ctrl_circuit_qubit]]
            orig_targ_server = placement.placement[circuit_qubit_to_vertex[orig_targ_circuit_qubit]]
            
            gate_server = placement.placement[gate_vertex]
            
            # If the original control circuit qubit has been placed on
            # the same server that the gate has been assigned to, then 
            # use the relabeled server qubit as the control
            if orig_ctrl_server == gate_server:
                new_ctrl = circuit_qubit_to_server_qubit[orig_ctrl_circuit_qubit]
            # If not, move the control to the ancilla qubits of the server
            # to which the gate has been assigned
            else:
                # new_register = server_to_register[gate_server]
                # new_ctrl = new_register[len(server_to_qubit_vertex_list[gate_server]) + orig_ctrl_server]
                new_ctrl = server_to_link_register[gate_server][orig_ctrl_server]
                
            # Similarly for the target qubit
            if orig_targ_server == gate_server:
                new_targ = circuit_qubit_to_server_qubit[orig_targ_circuit_qubit]
            else:
                # new_register = server_to_register[gate_server]
                # new_targ = new_register[len(server_to_qubit_vertex_list[gate_server]) + orig_targ_server]
                new_targ = server_to_link_register[gate_server][orig_targ_server]
            
            # Update the dictionary from gate vertices to commands to
            # include these new server qubits.
            gate_vertex_to_command[gate_vertex]['args'] = [new_ctrl, new_targ]
            
        print("gate_vertex_to_command", gate_vertex_to_command)
        
        new_command_list = []
        
        for gate_vertex, command in gate_vertex_to_command.items():
            
            gate_dict = {"vertex":gate_vertex, "op": command['command'].op, 'args':command['args'], 'role':'gate'}
            new_command_list.append(gate_dict)
                
        print("new_command_list", new_command_list)
        
        for edge in self.hyperedge_list:
            
            print("==============")
            print("edge", edge)
            
            gate_vertex_list = [vertex for vertex in edge['hyperedge'] if self.vertex_circuit_map[vertex]['type'] == 'gate']
            print("gate_vertex_list", gate_vertex_list)
            
            first_gate_found = False
            last_gate_found = False
            first_gate = 0
            last_gate = len(new_command_list) - 1
            
            while not first_gate_found:
                if new_command_list[first_gate]['role'] == 'gate':
                    if new_command_list[first_gate]['vertex'] in gate_vertex_list:
                        first_gate_found = True
                    else:
                        first_gate += 1
                else:
                    first_gate += 1
                
            while not last_gate_found:
                if new_command_list[last_gate]['role'] == 'gate':
                    if new_command_list[last_gate]['vertex'] in gate_vertex_list:
                        last_gate_found = True
                    else:
                        last_gate -= 1
                else:
                    last_gate -= 1
                
            print("first_gate", first_gate)
            print("last_gate", last_gate)
            
            unique_server_list = list(set([placement.placement[vertex] for vertex in edge['hyperedge'] if self.vertex_circuit_map[vertex]['type'] == 'gate']))
            print("unique_server_list", unique_server_list)
            
            qubit_list = [vertex for vertex in edge['hyperedge'] if self.vertex_circuit_map[vertex]['type'] == 'qubit']
            print("qubit_list", qubit_list)
            assert len(qubit_list) == 1
            qubit_vertex = qubit_list[0]
            
            qubit_server = placement.placement[qubit_vertex]
            print("qubit_server", qubit_server)
            
            for server in unique_server_list:
                print("server", server)
                if server == qubit_server:
                    continue
                else:
                    # new_command_list.insert(last_gate+1, {'role':'end', 'args':[server_to_register[server][len(server_to_qubit_vertex_list[server]) + qubit_server], server_to_register[qubit_server][0]]})
                    new_command_list.insert(last_gate+1, {'role':'end', 'args':[server_to_link_register[server][qubit_server], server_to_register[qubit_server][0]]})
            
            for server in unique_server_list:
                if server == qubit_server:
                    continue
                else:
                    # new_command_list.insert(first_gate, {'role':'start', 'args':[server_to_register[qubit_server][0], server_to_register[server][len(server_to_qubit_vertex_list[server]) + qubit_server]]})
                    new_command_list.insert(first_gate, {'role':'start', 'args':[server_to_register[qubit_server][0], server_to_link_register[server][qubit_server]]})
            
        print("new_command_list", new_command_list)

        for command in new_command_list:
            
            print("command", command)
            
            if command['role'] == 'gate':
                circ.add_gate(command['op'], command['args'])
            elif command['role'] == 'start':
                circ.add_custom_gate(start_proc, [], command['args'])
            elif command['role'] == 'end':
                circ.add_custom_gate(end_proc, [], command['args'])
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
