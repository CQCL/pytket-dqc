from pytket import Circuit
import numpy as np
from scipy.linalg import schur
from pytket.circuit import Unitary1qBox, Unitary2qBox, Op, OpType, Command, QubitRegister, Qubit
from pytket.transform import Transform
from warnings import warn
from networkx import from_dict_of_lists
from networkx.algorithms.bipartite import maximum_matching, to_vertex_cover

def is_global(command):
    """Boolean function that determines if a given two qubit command is global.

    :param command: Command which is to be checked.
    :type command: Command
    :return: If the command is global.
    :rtype: bool
    """
    if len(command.qubits) > 2:
        raise Exception("This command is not a one/two qubit command")
    
    return (len(command.qubits) > 1) and (command.qubits[0].reg_name != command.qubits[1].reg_name)

def is_diagonal(command):
    """Boolean function that determines if a given command has an associated matrix representation (in the computational basis) that is diagonal.
    This function uses the fastest answer presented here https://stackoverflow.com/questions/43884189/check-if-a-large-matrix-is-diagonal-matrix-in-python

    :param command: Command which is to be checked.
    :type command: Command
    :return: If the command is diagonal.
    :rtype: bool
    """
    if command.op.type == OpType.Rz: #::TODO There is probably a better way to handle symbolic gates..
        return True
    array = command.op.get_unitary()
    i, j = array.shape
    test = array.reshape(-1)[:-1].reshape(i-1, j+1)
    return ~np.any(test[:, 1:])

def is_antidiagonal(command):
    """Boolean function that determines if a given command has an associated matrix representation (in the computational basis) that is antidiagonal.
    This function uses the fastest answer presented here https://stackoverflow.com/questions/43884189/check-if-a-large-matrix-is-diagonal-matrix-in-python

    :param command: Command which is to be checked.
    :type command: Command
    :return: If the command is antidiagonal.
    :rtype: bool
    """

    array = np.flip(command.op.get_unitary(), 0) # Mirror the array
    i, j = array.shape
    test = array.reshape(-1)[:-1].reshape(i-1, j+1)
    return ~np.any(test[:, 1:])

def to_bipartite(circ, ancilla_limits = (-1, -1), simultaneous_max = False, debugging = False):
    """Generate a bipartite graph for a given circuit, with specific rules to determine which commands can be grouped together into a single vertex.

    :param circ: The circuit in question.
    :type circ: pytket.Circuit
    :param ancilla_limits: The limits on the number of ancilla on server 0 and server 1 respectively.
    :return: A bipartite graph representing the circuit
    :rtype: .BipartiteGraph

    ::TODO Test how it works on local two qubit gates::
    ::TODO Fix documentation::
    ::TODO Fix terrible naming schemes::
    """
    
    #initialise everything we need
    graph = {} #map from each vertex to all the vertices it is connected to
    
    servers = {
        circ.q_registers[0].name: BipartiteQReg(circ.q_registers[0], ancilla_limits[0], 'left'), #assign 0th server LHS
        circ.q_registers[1].name: BipartiteQReg(circ.q_registers[1], ancilla_limits[1], 'right') #assign 1st server RHS
    }

    left_qreg = servers[circ.q_registers[0].name]
    right_qreg = servers[circ.q_registers[1].name]
    
    current_vertex_index = 0 #tracks how many vertices have been made

    bipartite_qubits = {} #dictionary tracking all the qubits
    
    #create empty vertices for each qubit
    for qubit in circ.qubits:
        bipartite_qubit = BipartiteQubit(qubit)
        bipartite_qubits[qubit] = bipartite_qubit

    #count the number of commands acting on each qubit
    #TODO: look at efficiency of this - loop through circ.get_commands() twice which might not be very optimal
    #could maybe add each of the commands explicitly then remove as they are implemented?
    for cmd in circ.get_commands():
        for qubit in cmd.qubits:
            bipartite_qubits[qubit].commands.append(cmd)
    
    #create the vertices
    for cmd in circ.get_commands():
        q0 = bipartite_qubits[cmd.qubits[0]]
        server0 = servers[q0.reg_name]

        #Case: CZ
        if len(cmd.qubits) > 1:
            q1 = bipartite_qubits[cmd.qubits[1]]
            server1 = servers[q1.reg_name]

            #Case: Global CZ
            if is_global(cmd):
                #if there is no active vertex on q0, open one
                if q0.current_vertex == None:
                    if server1.is_full(): #close the most recently opened vertex on this server
                    #This is not optimal choice
                        server0.close_prev_vertex(server1)

                    q0.open_vertex(current_vertex_index, server0)
                    server1.add_to_packing(q0.current_vertex)
                    graph[q0.current_vertex] = set()
                    current_vertex_index += 1

                #elif there is an active vertex but we are only now adding global gates to it
                elif q0.current_vertex not in server1.currently_packing:
                    if server1.is_full():
                        server0.close_prev_vertex(server1)

                    server1.add_to_packing(q0.current_vertex)
                    graph[q0.current_vertex] = set()

                #repeat of the above if, elif statements
                if q1.current_vertex == None:
                    if server0.is_full():
                        server1.close_prev_vertex(server0)

                    q1.open_vertex(current_vertex_index, server1)
                    server0.add_to_packing(q1.current_vertex)
                    graph[q1.current_vertex] = set()
                    current_vertex_index += 1

                elif q1.current_vertex not in server0.currently_packing:
                    if server0.is_full():
                        server1.close_prev_vertex(server0)

                    graph[q1.current_vertex] = set()
                    server0.add_to_packing(q1.current_vertex)
                
                #connect the vertices in the graph
                graph[q0.current_vertex].add(q1.current_vertex)
                graph[q1.current_vertex].add(q0.current_vertex)

            #Case: Local CZ
            else:
                #open a vertex if none is open
                if q0.current_vertex == None:
                    q0.open_vertex(current_vertex_index, server0)
                    current_vertex_index += 1


                #repeat of the above if, elif statements
                if q1.current_vertex == None:
                    q1.open_vertex(current_vertex_index, server1)
                    current_vertex_index += 1
            
            #add the commands to their relevant vertices
            q0.add_cmd_to_current_vertex(cmd)
            q1.add_cmd_to_current_vertex(cmd)

            #close off vertex if there are no more commands on the qubit to pack
            if q0.commands_to_pack() == 0:
                if q0.current_vertex in server1.currently_packing:
                    server1.currently_packing.remove(q0.current_vertex)
                q0.close_vertex()

            # close off vertex if the next command is a non (anti) diagonal gate
            # fixes issues with memory count problems
            else:
                next_command = q0.next_command()
                if not(is_packable(next_command)):
                    if q0.current_vertex in server1.currently_packing:
                        server1.currently_packing.remove(q0.current_vertex)
                    q0.close_vertex()
                del next_command
            
            if q1.commands_to_pack() == 0:
                if q1.current_vertex in server0.currently_packing:
                    server0.currently_packing.remove(q1.current_vertex)
                q1.close_vertex()

            else:
                next_command = q1.next_command()
                if not is_packable(next_command):
                    if q1.current_vertex in server0.currently_packing:
                        server0.currently_packing.remove(q1.current_vertex)
                    q1.close_vertex()
                del next_command
                
        #Case: 1 qubit diagonal or antidiagonal gate -> can keep packing! Note this conditions is true also if the command is CZ, but if command is CZ earlier if should have triggered
        elif is_packable(cmd):
            #if there is no active vertex on this qubit, add one
            if q0.current_vertex == None:
                q0.open_vertex(current_vertex_index, server0)
                current_vertex_index += 1

            q0.add_cmd_to_current_vertex(cmd)

            if q0.commands_to_pack() == 0:
                server1 = [qreg for qreg in servers.values() if qreg != server0][0]
                if q0.current_vertex in server1.currently_packing:
                    server1.currently_packing.remove(q0.current_vertex)
                q0.close_vertex()

        #Case: a 1 qubit unitary that cannot be packed -> if the vertex is being packed, add the command and close the vertex, else open a new vertex
        else:
            if q0.current_vertex == None:
                q0.open_vertex(current_vertex_index, server0)
                current_vertex_index += 1

            q0.add_cmd_to_current_vertex(cmd)

            #end the vertex if it was being packed
            server1 = [qreg for qreg in servers.values() if qreg != server0][0]
            if q0.current_vertex in server1.currently_packing:
                server1.currently_packing.remove(q0.current_vertex)
                q0.close_vertex()

    # Explicit output if needed
    if debugging:
        for bpqubit in bipartite_qubits.values():
            print(f'server{bpqubit.reg_name} index{bpqubit.index}')
            for key in bpqubit.vertices.keys():
                print(f'Vertex {key}')
                print(bpqubit.vertices[key])
            print()
            print()

    #cleanse any vertices that aren't actually a part of the bipartite graph
    top_vertices = set()
    for vertex in left_qreg.vertices.keys():
        if vertex in graph.keys():
            top_vertices.add(vertex)

    bottom_vertices = set()
    for vertex in right_qreg.vertices.keys():
        if vertex in graph.keys():
            bottom_vertices.add(vertex)

    # Return the maximum number of simultaneous qubits needed
    if simultaneous_max:
        return BipartiteGraph(graph, top_vertices, bottom_vertices), (left_qreg.simultaneous_max, right_qreg.simultaneous_max)

    return BipartiteGraph(graph, top_vertices, bottom_vertices)

def cmd_to_CZ(command, circ):
    """Convert a two qubit controlled unitary gate into a controlled phase gate and local unitary gates.

    :param command: A two qubit controlled unitary.
    :type command: pytket.circuit.Command
    :param circ: The circuit to add the decomposition onto.
    :type circ: pytket.Circuit
    """
    #decompose controlled unitary box to CZ and local phase gates
    
    ctr = command.qubits[0]
    tar = command.qubits[1]
    
    #Notation from Matsui-san's master's thesis, renamed to alpha for consistency with tket conventions
    U = command.op.get_unitary()[[2,3],:][:, [2,3]] # only works if the command is a controlled unitary (i.e. do nothing if ctr is 0)
    D, W = schur(U, output = 'complex')

    H = 2**(-0.5) * np.matrix([
        [1, 1],
        [1, -1]
    ])

    I = np.identity(2)

    theta1 = np.angle(D[0][0])
    alpha1 = theta1 / np.pi
    
    theta2 = np.angle(D[1][1])
    alpha2 = theta2 / np.pi
    
    if alpha1 != 0.0: #don't bother if they rotation angle is 0
        circ.Rz(alpha1, ctr) #add local phase gate

    if W.all() == I.all() or W.all() == (-1 * I).all(): #don't do anything
        pass
    elif (W @ H).round(10).all() == I.all() or (W @ H).round(10).all() == (-1 * I).all(): #if W is Hadamard, then add that
        circ.H(tar)
    else:
        circ.add_unitary1qbox(Unitary1qBox(W), tar) #add W gate

    if alpha2 - alpha1 == 1.0: #if it's a Cz, add that
        circ.CZ(ctr, tar)
    else:
        circ.CRz(alpha2 - alpha1, ctr, tar) # else add arbitrary control phase gate

    if W.all() == I.all() or W.all() == (-1 * I).all():
        pass
    elif (W @ H).round(10).all() == I.all() or (W @ H).round(10).all() == (-1 * I).all(): #if W is Hadamard, then add that
        circ.H(tar)
    else:
        circ.add_unitary1qbox(Unitary1qBox(W.conj().T), tar) #add W gate
    return

def circ_to_CZ(circ, globals_only = True):
    """Given a circuit of local gates and controlled unitary gates, return an equivalent circuit that only utilises controlled phase gates as two qubit gates.

    :param circ: The circuit to be converted.
    :type circ: pytket.Circuit
    :return: The transformed circuit.
    :rtype: pytket.Circuit
    """
    #maybe better to just modify the existing circuit though?
    new_circ = Circuit()
    
    for q_reg in circ.q_registers: #recreate the original circuit size
        new_circ.add_q_register(q_reg.name, q_reg.size)
    
    for cmd in circ.get_commands():
        if (globals_only and is_global(cmd)) or (not globals_only and len(cmd.qubits) == 2):
            cmd_to_CZ(cmd, new_circ)

        else:
            new_circ.add_gate(cmd.op, cmd.args)
            
    return new_circ

def is_packable(command):
    return command.op.type == OpType.CZ or is_diagonal(command) or is_antidiagonal(command)

def preprocess(circuit): #convert the circuit to CZ and local gates, then merge together local gates where possible
        new_circuit = circ_to_CZ(circuit, globals_only=False)
        Transform.while_repeat(
            Transform.RemoveRedundancies(),
            Transform.CommuteThroughMultis()
            ).apply(new_circuit)
        return new_circuit

class BipartiteCircuit:

    def __init__(self, circuit):
        self.circuit = preprocess(circuit)
    
    def get_bipartite_graph(self, debugging = False):
        return to_bipartite(self.circuit, debugging = debugging)

    def minimum_vertex_cover(self):
        bipartite_graph = self.get_bipartite_graph()
        matching = maximum_matching(
            bipartite_graph.nx_graph,
            top_nodes = bipartite_graph.left_vertices
            )
        return to_vertex_cover(bipartite_graph.nx_graph, matching, bipartite_graph.left_vertices)

class BipartiteGraph:
    """A class that contains different descriptions of the same bipartite graph.
    """
    def __init__(self, full_graph, left_vertices, right_vertices):
        """Initialisation function.

        :param full_graph: The complete description of the graph, with all vertices given as keys mapping to the vertices that they are connected to.
        :type full_graph: dict{int: set(int)}
        :param left_vertices: The set of vertices in the 'left hand side' of the bipartite graph.
        :type left_vertices: set
        :param right_vertices: The set of vertices in the 'right hand side' of the bipartite graph.
        :type right_vertices: set
        """
        self.graph = full_graph
        self.left_vertices = left_vertices
        self.right_vertices = right_vertices
        self.nx_graph = {}
        for key, value in self.graph.items():
            self.nx_graph[key] = list(value)
        self.nx_graph = from_dict_of_lists(self.nx_graph)

class BipartiteQReg:
    # Quantum register extended to also contain the vertices that it contains in a bipartite graph
    def __init__(self, qubit_register, ancilla_limit, half):
        self.name = qubit_register.name
        self.size = qubit_register.size
        self.vertices = {} #map from a vertex on this server to the BipartiteQubit it corresponds to
        self.ancilla_limit = ancilla_limit #how many ancilla are allowed on this server
        self.half = half #which half of the bipartite graph does this register belong: 'left' or 'right'
        self.currently_packing = [] #list of vertices on the other server that are currently being packed TODO: Rethink this logic!!
        self.simultaneous_max = 0

    def is_full(self):
        if self.ancilla_limit == -1: #-1 corresponds to no limit on the number of ancilla being packed on this server
            return False
        elif self.ancilla_limit < len(self.currently_packing):
            raise ValueError(f'Too many vertices are being packed at once on server {self.name} - this is likely a bug with pytket-dqc.packing')
        return self.ancilla_limit == len(self.currently_packing)

    def close_prev_vertex(self, other_qreg, newly_opened_vertex = None):
        vertex_to_close = other_qreg.currently_packing[-1]
        other_qreg.currently_packing.remove(vertex_to_close)
        self.vertices[vertex_to_close].close_vertex() #TODO: naming is terrible here.. needs a rethink

    def add_to_packing(self, vertex_number):
        self.currently_packing.append(vertex_number)
        if self.simultaneous_max < len(self.currently_packing):
            self.simultaneous_max = len(self.currently_packing)

class BipartiteQubit:
    #Extension of qubit class to include the number of commands acting on it and information about bipartite graph vertices on this qubit
    def __init__(self, qubit):
        self.index = qubit.index
        self.reg_name = qubit.reg_name
        self.commands = []
        self.packed_commands = []
        self.is_packing = False
        self.current_vertex = None
        self.vertices = {} #map from a vertex index to a list of all the commands acting on it

    def open_vertex(self, new_vertex_number, qreg):
        if self.current_vertex != None:
            warn(f"A new vertex has been opened on qubit {self.name} before the previous one was closed. The previous vertex on this qubit has been closed (perhaps prematurely). Is this what you wanted?")
        self.current_vertex = new_vertex_number
        self.vertices[self.current_vertex] = []
        qreg.vertices[self.current_vertex] = self #add (key: value) pair (vertex: BipartiteQubit) to the quantum register

    def close_vertex(self):
        if self.current_vertex == None:
            warn(f"There wasn't a vertex open on qubit {self.reg_name} {self.index} in the first place. A possible error?")
        self.current_vertex = None

    def next_command(self):
        #get the next command that hasn't been included in a vertex yet on this qubit
        if len(self.packed_commands) == len(self.commands):
            raise Exception('All the commands have been packed')
        
        i = len(self.packed_commands) - len(self.commands)
        return self.commands[i]

    def commands_to_pack(self):
        return len(self.commands) - len(self.packed_commands)

    def add_cmd_to_current_vertex(self, cmd):
        self.vertices[self.current_vertex].append(cmd)
        self.packed_commands.append(cmd)