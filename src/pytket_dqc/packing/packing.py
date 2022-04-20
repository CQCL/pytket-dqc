from pytket import Circuit
from hopcroftkarp import HopcroftKarp as hk
import numpy as np
from scipy.linalg import schur
from pytket.circuit import Unitary1qBox, Unitary2qBox, Op, OpType, Command, QubitRegister, Qubit
from warnings import warn
import networkx as nx

def is_global(command):
    """Boolean function that determines if a given two qubit command is global.

    :param command: Command which is to be checked.
    :type command: Command
    :return: If the command is global.
    :rtype: bool
    """
    if len(command.qubits) != 2:
        raise Exception("This command is not a two qubit command")
    
    return (len(command.qubits) > 1) and (not command.qubits[0].reg_name == command.qubits[1].reg_name)

def is_diagonal(command):
    """Boolean function that determines if a given command has an associated matrix representation (in the computational basis) that is diagonal.
    This function uses the fastest answer presented here https://stackoverflow.com/questions/43884189/check-if-a-large-matrix-is-diagonal-matrix-in-python

    :param command: Command which is to be checked.
    :type command: Command
    :return: If the command is diagonal.
    :rtype: bool
    """

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

def unmatched_vertices(graph, matching, left_only = True):
    """Given a graph and a matching, return a list of the vertices in the graph that are not in the matching. By default, only returns the vertices in the 'left hand side' of the bipartite graph (this is what is needed for finding minimum vertex cover).

    :param graph: The graph of interest.
    :type graph: .BipartiteGraph
    :param matching: Maps between vertices representing edges in the matching.
    :type matching: dict{int: int}
    :param left_only: If True, only the vertices in the LHS of the graph are returned.
    :type left_only: bool
    :return: The vertices in the graph that are not in the matching.
    :rtype: set
    """

    vertices = set()
    for vertex in graph.full_graph.keys():
        if vertex not in matching.keys():
            vertices.add(vertex)
    n_vertices = len(graph.full_graph.keys())
    
    if left_only:
        vertices = vertices & graph.left_vertices
    
    return vertices

def connected_by_alternating_path(graph, vertex, matching):
    """Given a matching for a bipartite graph and a vertex not in the matching of that graph, return a set of vertices that are connected to the given vertex by a path that alternates between edges in the matching and not in the matching.

    :param graph: The graph of interest.
    :type graph: .BipartiteGraph
    :param vertex: The vertex to start the alternating paths from.
    :type vertex: int
    :param matching: All the edges of the graph in the matching.
    :type matching: dict{int: int}
    :return: All the vertices connected to the starting vertex by an alternating path.
    :rtype: set
    """
    
    vertices_to_check = {vertex}
    checked_vertices = set()
    vertices = {vertex}
    check_in_matching = True
    
    while bool(vertices_to_check): #while the set is non-empty
        next_vertices_to_check = set()
        for current_vertex in vertices_to_check:
            connected = connected_vertices(graph, current_vertex, matching, not check_in_matching)
            next_vertices_to_check.update(connected)
            vertices.update(connected)
            checked_vertices.add(current_vertex)
        vertices_to_check = next_vertices_to_check - checked_vertices
        check_in_matching = not check_in_matching
    
    return vertices
        
def connected_vertices(graph, vertex, matching, check_in_matching, include_original = False):
    """Finds all vertices directly connected to the given vertex by edges that are/aren't in the matching.

    :param graph: The bipartite graph in question.
    :type graph: .BipartiteGraph
    :param vertex: The vertex to search from.
    :type vertex: int
    :param matching: A matching on the bipartite graph.
    :type matching: dict{int: int}
    :param check_in_matching: If True, search for vertices connected to given vertex by an edge that is in the matching.
    :type check_in_matching: bool
    :param include_original: Include the given vertex in the resulting set of vertices, defaults to False
    :type include_original: bool, optional
    :return: All the vertices connected to the given vertex by edges that are/aren't in the matching.
    :rtype: set
    """

    connected = graph.full_graph[vertex]
    vertices = set()
    for connected_vertex in connected:
        if check_in_matching == in_matching(matching, (vertex, connected_vertex)):
            vertices.add(connected_vertex)
            
    if include_original:
        vertices.add(vertex)
    
    return vertices
    
def in_matching(matching, edge):
    """Is a given edge in a given matching.

    :param matching: The matching to check against.
    :type matching: dict{int: int}
    :param edge: The edge of the graph to query with. Assumed the edge is pulled from graph.full_graph.items, hence tuple.
    :type edge: tuple
    :return: Is the given edge in the matching.
    :rtype: bool
    """
    vertex = edge[0]
    return (vertex in matching.keys()) and (matching[vertex] == edge[1]) #this may break if short circuiting is removed at somepoint

def minimum_vertex_cover(graph):
    """Find a minimum vertex cover of a graph. The notation and algorithm is taken from https://en.wikipedia.org/wiki/K%C5%91nig%27s_theorem_(graph_theory)

    :param graph: The graph in question.
    :type graph: .BipartiteGraph
    :return: A minimum vertex cover.
    :rtype: set
    """

    L = graph.left_vertices
    R = graph.right_vertices
    M = hk(graph.edges_from_left_only()).maximum_matching()
    U = unmatched_vertices(graph, M, L)
    Z = set()
    Z.update(U)
    n_vertices = len(graph.full_graph)
    
    #find all the vertices connected by an alternating path
    for vertex in U:
        connected = connected_by_alternating_path(graph, vertex, M)
        Z.update(connected)

    return ((L - Z) | (R & Z))

def to_bipartite(circ, ancilla_limits = (-1, -1), debugging = False):
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

    bipartite_qubits = {}
    
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

        #Case: global command
        if len(cmd.qubits) > 1 and is_global(cmd):
            q1 = bipartite_qubits[cmd.qubits[1]]
            server1 = servers[q1.reg_name]

            #if there is no active vertex on q0, open one
            if q0.current_vertex == None:
                if server1.is_full(): #close the most recently opened vertex on this server
                #This is not optimal choice
                    server0.close_prev_vertex(server1)

                q0.open_vertex(current_vertex_index, server0)
                server1.currently_packing.append(q0.current_vertex)
                graph[q0.current_vertex] = set()
                current_vertex_index += 1

            #elif there is an active vertex but we are only now adding global gates to it
            elif q0.current_vertex not in server1.currently_packing:
                if server1.is_full():
                    server0.close_prev_vertex(server1)

                server1.currently_packing.append(q0.current_vertex)
                graph[q0.current_vertex] = set()

            #repeat of the above if, elif statements
            if q1.current_vertex == None:
                if server0.is_full():
                    server1.close_prev_vertex(server0)

                q1.open_vertex(current_vertex_index, server1)
                server0.currently_packing.append(q1.current_vertex)
                graph[q1.current_vertex] = set()
                current_vertex_index += 1

            elif q1.current_vertex not in server0.currently_packing:
                if server0.is_full():
                    server1.close_prev_vertex(server0)

                graph[q1.current_vertex] = set()
                server0.currently_packing.append(q1.current_vertex)
            
            #connect the vertices in the graph
            graph[q0.current_vertex].add(q1.current_vertex)
            graph[q1.current_vertex].add(q0.current_vertex)

            #add the commands to their relevant vertices
            q0.add_cmd_to_current_vertex(cmd)
            q1.add_cmd_to_current_vertex(cmd)

            #close off vertex if there are no more commands on the qubit to pack
            if q0.commands_to_pack() == 0:
                server1.currently_packing.remove(q0.current_vertex)
                q0.close_vertex()
            
            if q1.commands_to_pack() == 0:
                server0.currently_packing.remove(q1.current_vertex)
                q1.close_vertex()
        
        #Case: diagonal or antidiagonal gate -> can keep packing!
        elif is_diagonal(cmd) or is_antidiagonal(cmd):
            #if there is no active vertex on this qubit, add one
            if q0.current_vertex == None:
                q0.open_vertex(current_vertex_index, server0)
                current_vertex_index += 1

            q0.add_cmd_to_current_vertex(cmd)
            if q0.commands_to_pack() == 0:
                server1 = [qreg for qreg in servers.values() if qreg != server0][0]
                server1.currently_packing.remove(q0.current_vertex) #the other qreg!!
                q0.close_vertex()

        #Case: a local unitary that cannot be packed -> if the vertex is being packed, add the command and close the vertex, else open a new vertex
        else:
            if q0.current_vertex == None:
                q0.open_vertex(current_vertex_index, server0)
                current_vertex_index += 1

            q0.add_cmd_to_current_vertex(cmd)
            server1 = [qreg for qreg in servers.values() if qreg != server0][0]

            #end the vertex if it was being packed
            if q0.current_vertex in server1.currently_packing:
                server1.currently_packing.remove(q0.current_vertex)
                q0.close_vertex()

    if debugging:
        for bpqubit in bipartite_qubits.values():
            print(f'server{bpqubit.reg_name} index{bpqubit.index}')
            for key in bpqubit.vertices.keys():
                print(f'Vertex {key}')
                print(bpqubit.vertices[key])
            print()
            print()
    return BipartiteGraph(graph, set(left_qreg.vertices.keys()), set(right_qreg.vertices.keys()))

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
    U = command.op.get_unitary()[[2,3],:][:, [2,3]]
    D, W = schur(U, output = 'complex')
    theta1 = np.angle(D[0][0])
    alpha1 = theta1 / np.pi
    
    theta2 = np.angle(D[1][1])
    alpha2 = theta2 / np.pi
    
    circ.Rz(alpha1, ctr) #add local phase gate
    circ.add_unitary1qbox(Unitary1qBox(W), tar) #add W gate
    circ.CRz(alpha2 - alpha1, ctr, tar) #add control phase gate
    circ.add_unitary1qbox(Unitary1qBox(W.conj().T), tar) #add gate to complete decomposition
    
    return

def circ_to_CZ(circ):
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
        if is_global(cmd):
            cmd_to_CZ(cmd, new_circ)
            
        else:
            new_circ.add_gate(cmd.op, cmd.args)
            
    return new_circ

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
        self.full_graph = full_graph
        self.left_vertices = left_vertices
        self.right_vertices = right_vertices
        self.nx_graph = {}
        for key, value in self.full_graph.items():
            self.nx_graph[key] = list(value)
        self.nx_graph = nx.from_dict_of_lists(self.nx_graph)
    
    def edges_from_left_only(self):
        """Return an equivalent bipartite graph with only the left vertices as keys, listing the right vertices they are connected to.

        :return: The bipartite graph with only left vertices as keys
        :rtype: dict{int: set(int)}
        """
        left_only = self.full_graph.copy()
        for vertex in self.right_vertices:
            del left_only[vertex]
        return left_only

    def edges_from_right_only(self):
        """Return an equivalent bipartite graph with only the right vertices as keys, listing the left vertices they are connected to.

        :return: The bipartite graph with only right vertices as keys
        :rtype: dict{int: set(int)}
        """
        right_only = self.full_graph.copy()
        for vertex in self.left_vertices:
            del right_only[vertex]
        return right_only

class BipartiteQReg:
    # Quantum register extended to also contain the vertices that it contains in a bipartite graph
    def __init__(self, qubit_register, ancilla_limit, half):
        self.name = qubit_register.name
        self.size = qubit_register.size
        self.vertices = {} #map from a vertex on this server to the BipartiteQubit it corresponds to
        self.ancilla_limit = ancilla_limit #how many ancilla are allowed on this server
        self.half = half #which half of the bipartite graph does this register belong: 'left' or 'right'
        self.currently_packing = [] #list of vertices on the other server that are currently being packed TODO: Rethink this logic!!

    def is_full(self):
        if self.ancilla_limit == -1: #-1 corresponds to no limit on the number of ancilla being packed on this server
            return False
        elif self.ancilla_limit < len(self.currently_packing):
            raise ValueError(f'For some reason, too many vertices are being packed at once on server {self.name}')
        return self.ancilla_limit == len(self.currently_packing)

    def close_prev_vertex(self, other_qreg, newly_opened_vertex = None):
        vertex_to_close = other_qreg.currently_packing[-1]
        other_qreg.currently_packing.remove(vertex_to_close)
        self.vertices[vertex_to_close].close_vertex() #TODO: naming is terrible here.. needs a rethink

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

    def get_packed_commands(self):
        packed_commands = []
        for packed_commands_list in self.vertices.values():
            packed_commands += packed_commands_list
        return packed_commands

    def commands_to_pack(self):
        return len(self.commands) - len(self.get_packed_commands())

    def add_cmd_to_current_vertex(self, cmd):
        self.vertices[self.current_vertex].append(cmd)
