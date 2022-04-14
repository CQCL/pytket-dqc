from pytket import Circuit
from hopcroftkarp import HopcroftKarp as hk
import numpy as np
from scipy.linalg import schur
from pytket.circuit import Unitary1qBox, Unitary2qBox, Op, OpType, Command

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
    in_matching = True
    
    while bool(vertices_to_check): #while the set is non-empty
        next_vertices_to_check = set()
        for current_vertex in vertices_to_check:
            connected = connected_vertices(graph, current_vertex, matching, not in_matching)
            next_vertices_to_check.update(connected)
            vertices.update(connected)
            checked_vertices.add(current_vertex)
        vertices_to_check = next_vertices_to_check - checked_vertices
        in_matching = not in_matching
    
    return vertices
        
def connected_vertices(graph, vertex, matching, in_matching, include_original = False):
    """Finds all vertices directly connected to the given vertex by edges that are/aren't in the matching.

    :param graph: The bipartite graph in question.
    :type graph: .BipartiteGraph
    :param vertex: The vertex to search from.
    :type vertex: int
    :param matching: A matching on the bipartite graph.
    :type matching: dict{int: int}
    :param in_matching: If True, search for vertices connected to given vertex by an edge that is in the matching.
    :type in_matching: bool
    :param include_original: Include the given vertex in the resulting set of vertices, defaults to False
    :type include_original: bool, optional
    :return: All the vertices connected to the given vertex by edges that are/aren't in the matching.
    :rtype: set
    """

    connected = graph.full_graph[vertex]
    vertices = set()
    for connected_vertex in connected:
        if in_matching == in_matching(matching, (vertex, connected_vertex)):
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

def to_bipartite(circ):
    """Generate a bipartite graph for a given circuit, with specific rules to determine which commands can be grouped together into a single vertex.

    :param circ: The circuit in question.
    :type circ: pytket.Circuit
    :return: A bipartite grpah representing the circuit
    :rtype: .BipartiteGraph
    """
    
    #initialise everything we need
    graph = {} #map from each vertex to all the vertices it is connected to
    vertices = {} #map from the qubit to vertex information
    
    left_server = circ.q_registers[0] #LHS of the bipartite graph
    right_server = circ.q_registers[1] #RHS of the bipartite graph
    left_vertices = set() #set of 'left' vertices in the bipartite graph
    right_vertices = set() #set of 'right' vertices in the bipartite graph
    
    current_vertex_index = 0 #tracks how many vertices have been made
    
    #create empty vertices for each qubit
    for qubit in circ.qubits:
        vertices[qubit] = {
            "current_vertex": current_vertex_index, #the current working vertex of this qubit
            "all_vertex_indices": set([current_vertex_index]), #the set of all vertices associated with the qubit
            "vertex_gates": {current_vertex_index: []}, #a map from a vertex index to all commands in the vertex
        }
        if qubit.reg_name == left_server.name:
            left_vertices.add(current_vertex_index)
        else:
            right_vertices.add(current_vertex_index)
        graph[current_vertex_index] = set()
        current_vertex_index += 1
    
    for cmd in circ.get_commands():
        q0 = cmd.qubits[0]
        q0_current_vertex = vertices[q0]["current_vertex"]
        if len(cmd.qubits) > 1 and is_global(cmd):
            q1 = cmd.qubits[1]
            q1_current_vertex = vertices[q1]["current_vertex"]
            
            #add the cmds to the working vertex
            vertices[q0]["vertex_gates"][q0_current_vertex].append(cmd)
            vertices[q1]["vertex_gates"][q1_current_vertex].append(cmd)
            
            #connect the vertices in the graph
            graph[q0_current_vertex].add(q1_current_vertex)
            graph[q1_current_vertex].add(q0_current_vertex)
            
        elif is_diagonal(cmd) or is_antidiagonal(cmd):
            vertices[q0]["vertex_gates"][q0_current_vertex].append(cmd)
            
        else: #else cannot be packed, so we need a new vertex for this qubit
            vertices[q0]["current_vertex"] = current_vertex_index
            vertices[q0]["all_vertex_indices"].add(current_vertex_index)
            vertices[q0]["vertex_gates"][current_vertex_index] = []
            if q0.reg_name == left_server.name:
                left_vertices.add(current_vertex_index)
            else:
                right_vertices.add(current_vertex_index)
            graph[current_vertex_index] = set()
            current_vertex_index += 1

    return BipartiteGraph(graph, left_vertices, right_vertices), vertices

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