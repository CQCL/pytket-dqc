from pytket import Circuit
import numpy as np
from pytket.circuit import Op, OpType, Command, QubitRegister, Qubit # type: ignore
from pytket_dqc.utils.op_analysis import is_diagonal, is_antidiagonal, get_qubit_reg_num
from pytket_dqc.utils.gateset import start_proc, end_proc, dqc_gateset_predicate
from warnings import warn
from networkx import from_dict_of_lists, Graph # type: ignore
from networkx.algorithms.bipartite import maximum_matching, to_vertex_cover # type: ignore

def pack_circuit(bipartite_circuit):
    """Create a circuit from a BipartiteCircuit which has all the StartingProcesses and EndingProcesses included in the circuit.

    :param bipartite_circuit: The BipartiteCircuit from which this circuit is to be constructed.
    :type bipartite_circuit: BipartiteCircuit
    :return: The constructed circuit.
    :rtype: pytket.Circuit
    """
    circuit = Circuit()

    for extended_command in bipartite_circuit.extended_commands:
        for qubit in extended_command.command.qubits:
            if qubit not in circuit.qubits:
                circuit.add_qubit(qubit)

        add_extended_command_to_circuit(circuit, extended_command, bipartite_circuit)

    return circuit


def add_extended_command_to_circuit(circuit, extended_command, bipartite_circuit):
    """Add an extended command to a given circuit.

    If needs be, also adds starting and ending processes to the circuit, as dictated by the supplied BipartiteCircuit.mvc.
    Note that it is assumed that the global gates are CPhase gates.

    TODO Consider the case where we have global CX in the case of qubit swaps e.g. in routing.

    :param circuit: The circuit to which the extended command is to be appended.
    :type circuit: pytket.Circuit
    :param extended_command: The extended command to append.
    :type extended_command: ExtendedCommand
    :param bipartite_circuit: The bipartite circuit associated with the extended commands.
    :type bipartite_circuit: BipartiteCircuit
    """
    # Add an extended command (EC) to a circuit, with packing if needed as given by the supplied mvc
    # Cases are as follows
    # 1) The EC is 1q or not packable are no packings on the qubit. -> Just add it to circuit
    # 2) The EC is not packable but there are packings on the qubit which need to be terminated. -> Terminate all packings, add the EC to circuit
    # 4) The EC is packable as a 1q diagonal -> Just add it to circuit
    # 5) The EC is packable as a 1q antidiagonal -> Add to circuit and add correctional X gate to all current packings
    # 6) The EC is packable as a 2q CZ -> Add the CZ to the packing on the appropriate l_qubit.

    # Handle all the cases where the ExtendedCommand is local.
    if extended_command.is_local():
        vertex = extended_command.vertices[0]
        extended_qubit = vertex.extended_qubit
        if (
            not extended_command.is_packable()
        ):  # If not packable, we need to close off all current packings on this qubit
            while len(extended_qubit.current_l_qubits) > 0:
                l_qubit = extended_qubit.current_l_qubits[0]
                l_qubit.stop_packing()
                circuit.add_custom_gate(
                    end_proc, [], [l_qubit.qubit, l_qubit.origin_extended_qubit.qubit]
                )

        elif extended_qubit.is_packing() and is_antidiagonal(
            extended_command.command.op
        ):
            for l_qubit in extended_qubit.current_l_qubits:
                circuit.X(l_qubit.qubit)

        circuit.add_gate(extended_command.command.op, extended_command.command.args)
        vertex.added_extended_commands.append(extended_command)

    else:  # Handle cases of a global CZ
        if extended_command.command.op.type == OpType.CX:
            raise Exception("Global CX gates are not allowed.")
        for vertex in extended_command.vertices:  # Add starting processes if needed
            if vertex.i in bipartite_circuit.mvc and not vertex.is_packing:
                le_reg_num = vertex.get_connected_server_reg_num()
                le_reg_name = (
                    f"Server {le_reg_num} Link Edge {bipartite_circuit.link_edge_count}"
                )
                bipartite_circuit.link_edge_count += 1
                le_q_reg = QubitRegister(le_reg_name, 1)
                circuit.add_q_register(le_q_reg)
                link_qubit = LinkQubit(le_q_reg[0], vertex)
                link_qubit.start_packing()
                circuit.add_custom_gate(
                    start_proc,
                    [],
                    [link_qubit.origin_extended_qubit.qubit, link_qubit.qubit],
                )

        packed_vertex = [
            vertex
            for vertex in extended_command.vertices
            if vertex.i in bipartite_circuit.mvc
        ][
            0
        ]  # If it's a global CZ, one of the two vertices it belongs to must be in the MVC, so select that vertex
        l_qubit = packed_vertex.link_qubit
        qubit = [
            qubit
            for qubit in extended_command.command.qubits
            if qubit != l_qubit.origin_extended_qubit.qubit
        ][0]
        circuit.add_gate(extended_command.command.op, [qubit, l_qubit.qubit])
        for vertex in extended_command.vertices:
            vertex.added_extended_commands.append(extended_command)

    for vertex in extended_command.vertices:
        if (
            vertex.i in bipartite_circuit.mvc
            and vertex.extended_commands == vertex.added_extended_commands
        ):  # If all of the commands on the vertex are packed we add an ending process
            l_qubit = vertex.link_qubit
            l_qubit.stop_packing()
            circuit.add_custom_gate(
                end_proc, [], [l_qubit.qubit, l_qubit.origin_extended_qubit.qubit]
            )


def add_edge_to_graph(vertex1, vertex2, graph):
    """Given two CommandVertexs and a graph, add an edge to the graph that connects the two vertices.
    If either of the vertices are not yet on the graph then also add them to the graph.

    :param vertex1: The first vertex we wish to add an edge between.
    :type vertex1: CommandVertex
    :param vertex2: The second vertex we wish to add an edge between.
    :type vertex2: CommandVertex
    :param graph: The graph to which the edge (and vertices) should be added to.
    :type graph: networkx.Graph
    """
    if not (
        vertex1.is_on_graph or vertex2.is_on_graph
    ):  # Neither vertices are on the graph, so can fix which half of the graph they are on.
        vertex1.set_bipartite_i(0)
        vertex2.set_bipartite_i(1)
        graph.add_nodes_from([vertex1.get_i()], bipartite=vertex1.get_bipartite_i())
        graph.add_nodes_from([vertex2.get_i()], bipartite=vertex2.get_bipartite_i())

    elif vertex1.is_on_graph:
        vertex2.set_bipartite_i(
            [i for i in range(2) if i != vertex1.get_bipartite_i()][0]
        )
        graph.add_nodes_from([vertex2.get_i()], bipartite=vertex2.get_bipartite_i())

    elif vertex2.is_on_graph:
        vertex1.set_bipartite_i(
            [i for i in range(2) if i != vertex2.get_bipartite_i()][0]
        )
        graph.add_nodes_from([vertex1.get_i()], bipartite=vertex1.get_bipartite_i())

    graph.add_edges_from([(vertex1.get_i(), vertex2.get_i())])


def to_extended_commands(commands):
    """Given a list of commands, convert them to ExtendedCommands.

    Typically this list of commands comes from pytket.circuit.Circuit.get_commands(). Their index is given by the order in the list they occur at.

    :param commands: A list of commands
    :type commands: List[pytket.circuit.Command]
    :return: A list of ExtendedCommands
    :rtype: List[ExtendedCommand]
    """
    return [ExtendedCommand(i, command) for i, command in enumerate(commands)]


class LinkQubit:
    def __init__(self, qubit, vertex):
        self.qubit = qubit
        self.origin_extended_qubit = vertex.extended_qubit
        self.vertex = vertex

    def start_packing(self):
        self.is_open = True
        self.vertex.is_packing = True
        self.vertex.link_qubit = self
        self.origin_extended_qubit.start_packing(self)

    def stop_packing(self):
        self.is_open = False
        self.vertex.is_packing = False
        self.origin_extended_qubit.stop_packing(self)


class CommandVertex:
    def __init__(self, i, extended_qubit):
        self.i = i
        self.extended_qubit = extended_qubit
        self.reg_num = get_qubit_reg_num(extended_qubit.qubit)
        self.extended_commands = []
        self.added_extended_commands = []
        self.is_open = False
        self.connected_server_reg_num = None
        self.connected_vertices = set()
        self.is_on_graph = False
        self.is_packing = False
        self.link_qubit = None
        extended_qubit.add_vertex(self)

    def close(self):
        self.is_open = False

    def get_connected_server_reg_num(self):
        return self.connected_server_reg_num

    def is_connected(self):
        return self.connected_server_reg_num is not None

    def get_i(self):
        return self.i

    def get_reg_num(self):
        return self.reg_num

    def add_extended_command(self, extended_command):
        self.extended_commands.append(extended_command)

    def connect_to_server(self, connected_server_reg_num):
        self.connected_server_reg_num = connected_server_reg_num
        self.is_open = True

    def connect_vertex(self, vertex):
        self.connected_vertices.add(vertex)
        assert vertex.get_reg_num() == self.connected_server_reg_num
        self.is_open = True

    def get_extended_command_indices(self):
        indices = []
        for extended_command in self.extended_commands:
            indices.append(extended_command.i)
        return indices

    def set_bipartite_i(
        self, i
    ):  # The bipartite partition on which this vertex resides
        self.bipartite_i = i
        self.is_on_graph = True

    def get_bipartite_i(self):
        return self.bipartite_i


class ExtendedQubit:
    def __init__(self, qubit, extended_commands):
        self.qubit = qubit
        self.extended_commands = extended_commands
        self.vertices = []
        self.last_used_vertex = None
        self.all_l_qubits = []
        self.current_l_qubits = []

    def is_packing(self):
        return len(self.current_l_qubits) > 0

    def start_packing(self, l_qubit):
        self.all_l_qubits.append(l_qubit)
        self.current_l_qubits.append(l_qubit)

    def stop_packing(self, l_qubit):
        self.current_l_qubits.remove(l_qubit)

    def get_extended_commands(self):
        return self.extended_commands

    def add_extended_command(self, extended_command):
        self.extended_commands.append(extended_command)

    def add_vertex(self, vertex):
        self.vertices.append(vertex)
        self.last_used_vertex = self.vertices[-1]

    def get_currently_connected_vertices(self):
        # a set of vertices on this qubit that are connected and open
        connected_vertices = set()
        for vertex in self.vertices:
            if vertex.is_open and vertex.is_connected():
                connected_vertices.add(vertex)
        return connected_vertices

    def get_currently_connected_servers(self):
        # A connected server is one from which this ExtendedQubit has a StartingProcess leading to a Link Edge qubit on that server
        # A currently connected server is a connected server for which the StartingProcess has not been ended via an EndingProcess
        # In essence this function finds every server that this qubit is connected to
        # using the server number as keys to a list of vertices that are connected to the relevant Link Edge qubit.
        connected_servers_dict = (
            {}
        )  # Maps the connected_server_reg_num -> all vertices connected to this vertex on that register
        connected_vertices = self.get_currently_connected_vertices()
        for vertex in connected_vertices:
            if vertex.get_connected_server_reg_num() in connected_servers_dict.keys():
                connected_servers_dict[vertex.get_connected_server_reg_num()].add(
                    vertex
                )
            else:
                connected_servers_dict[vertex.get_connected_server_reg_num()] = {vertex}

        return connected_servers_dict

    def get_command_indices(self):
        command_indices = []
        for extended_command in self.get_extended_commands():
            command_indices.append(extended_command.get_i())
        return command_indices

    def set_last_used_vertex(self, vertex):
        self.last_used_vertex = vertex

    def close_all_vertices(self):
        for vertex in self.vertices:
            vertex.close()


class ExtendedCommand:
    def __init__(self, i, command):
        self.i = i
        self.command = command
        self.vertices = []

    def add_vertex(self, vertex):
        self.vertices.append(vertex)

    def get_vertices(self):
        return self.vertices

    def is_1q(self):
        return len(self.command.qubits) == 1

    def other_arg_qubit(self, this_qubit):
        return [
            arg_qubit for arg_qubit in self.command.qubits if arg_qubit != this_qubit
        ][0]

    def other_arg_server_num(self, this_qubit):
        return get_qubit_reg_num(self.other_arg_qubit(this_qubit))

    def is_local(self):
        if self.is_1q():
            return True
        q0 = self.command.qubits[0]
        return get_qubit_reg_num(q0) == self.other_arg_server_num(q0)

    def get_i(self):
        return self.i

    def is_1q_packable(self):
        return self.is_1q() and self.is_packable()

    def is_packable(self):
        return is_diagonal(self.command.op) or is_antidiagonal(self.command.op)


class BipartiteCircuit:
    def __init__(self, circuit):
        if not dqc_gateset_predicate.verify(circuit):
            raise Exception('The given circuit is not in the allowed gateset')
        self.circuit = circuit
        self.graph, self.extended_commands = self.to_bipartite()
        self.link_edge_count = 0
        self.top_nodes = {
            n for n, d in self.graph.nodes(data=True) if d["bipartite"] == 0
        }
        self.matching = maximum_matching(self.graph, self.top_nodes)
        self.mvc = to_vertex_cover(self.graph, self.matching, self.top_nodes)
        self.packed_circuit = pack_circuit(self)

    def get_graph(self):
        return self.graph

    def get_packed_circuit(self):
        return self.packed_circuit

    def get_ebit_cost(self):
        return len(self.mvc)

    def to_bipartite(self):
        """Given a circuit, create a bipartite graph representing that circuit.

        Currently assumes the gateset only has CZ as two qubit gates. The circuit must also have been placed onto servers.

        TODO raise exception if this is not the case?

        :return graph: The graph representation of the circuit.
        :rtype: networkx.Graph
        """
        extended_commands = to_extended_commands(
            self.circuit.get_commands()
        )  # Give each command an index to reference it by.
        next_vertex_index = 0
        graph = Graph()

        # Convert qubits to ExtendedQubits (qubits with some extra functionality). extended_qubits maps each qubit -> its ExtendedQubit
        extended_qubits = {}
        for qubit in self.circuit.qubits:
            extended_qubits[qubit] = ExtendedQubit(qubit, [])
            CommandVertex(next_vertex_index, extended_qubits[qubit]) # ALTER BEHAVIOUR SO THIS IS METHOD OF EXTENDED_QUBIT
            next_vertex_index += 1

        # Populate the ExtendedCommands list for each ExtendedQubit
        for extended_command in extended_commands:
            for qubit in extended_command.command.qubits:
                extended_qubits[qubit].add_extended_command(extended_command)

        # Create all the vertices on each qubit
        for extended_qubit in extended_qubits.values():
            for extended_command in extended_qubit.get_extended_commands():
                if (
                    extended_command.is_packable() and extended_command.is_local()
                ):  # Local packable gate
                    vertex = extended_qubit.last_used_vertex
                    extended_command.add_vertex(vertex)
                    vertex.add_extended_command(extended_command)

                elif extended_command.is_1q():  # 1 qubit non-(anti)diagonal gate
                    extended_qubit.close_all_vertices()
                    new_vertex = CommandVertex(next_vertex_index, extended_qubit)
                    extended_command.add_vertex(new_vertex)
                    new_vertex.add_extended_command(extended_command)
                    extended_qubit.add_vertex(new_vertex)
                    next_vertex_index += 1

                else:  # Non-local CZ gate
                    currently_connected_servers = (
                        extended_qubit.get_currently_connected_servers()
                    )
                    if (
                        extended_command.other_arg_server_num(extended_qubit.qubit)
                        in currently_connected_servers.keys()
                    ):
                        # We are currently connected from this qubit to the server of the other qubit for this CZ, so we can add this CZ to this packing set.
                        vertex = list(
                            currently_connected_servers[
                                extended_command.other_arg_server_num(extended_qubit.qubit)
                            ]
                        )[0]
                        extended_qubit.add_vertex(vertex)

                    else:  # Must create a new connection from this qubit to the server.
                        if extended_qubit.last_used_vertex.is_connected():
                            vertex = CommandVertex(next_vertex_index, extended_qubit)
                            next_vertex_index += 1
                            extended_qubit.add_vertex(vertex)
                        else:
                            vertex = extended_qubit.last_used_vertex
                        other_server_reg_num = extended_command.other_arg_server_num(
                            extended_qubit.qubit
                        )
                        vertex.connect_to_server(other_server_reg_num)

                    extended_command.add_vertex(vertex)
                    vertex.add_extended_command(extended_command)

                    if (
                        len(extended_command.get_vertices()) == 2
                    ):  # Once both the vertices for the CZ are added, we can add the edge to the graph
                        other_vertex = [
                            other_vertex
                            for other_vertex in extended_command.get_vertices()
                            if other_vertex != vertex
                        ][0]
                        add_edge_to_graph(vertex, other_vertex, graph)

        return graph, extended_commands

