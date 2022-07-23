from pytket import Circuit
from pytket.circuit import (  # type: ignore
    OpType,
    QubitRegister,
)
from networkx import Graph  # type: ignore
from networkx.algorithms.bipartite import (  # type: ignore
    maximum_matching,
    to_vertex_cover,
)
from pytket_dqc.utils.op_analysis import (
    is_diagonal,
    is_antidiagonal,
)
from pytket_dqc.utils.gateset import (
    start_proc,
    end_proc,
    dqc_gateset_predicate,
)
from pytket_dqc.distributors import Random
from pytket_dqc.circuits import DistributedCircuit


def pack_circuit(bipartite_circuit):
    """Create a circuit from a BipartiteCircuit
    including StartingProcesses and EndingProcesses.

    :param bipartite_circuit: The BipartiteCircuit
        from which this circuit is to be constructed.
    :type bipartite_circuit: BipartiteCircuit
    :return: The constructed circuit.
    :rtype: pytket.Circuit
    """
    circuit = Circuit()

    for extended_command in bipartite_circuit.extended_commands:
        for qubit in extended_command.command.qubits:
            if qubit not in circuit.qubits:
                circuit.add_qubit(qubit)

        add_extended_command_to_circuit(
            circuit,
            extended_command,
            bipartite_circuit,
        )

    return circuit


def add_extended_command_to_circuit(
    circuit, extended_command, bipartite_circuit
):
    """Add an extended command to a given circuit.

    If needs be, also adds starting and ending processes to the circuit,
    as dictated by the supplied BipartiteCircuit.mvc.
    Note that it is assumed that the global gates are CPhase gates.

    :param circuit: The circuit to which the
        extended command is to be appended.
    :type circuit: pytket.Circuit
    :param extended_command: The extended command to append.
    :type extended_command: ExtendedCommand
    :param bipartite_circuit: The bipartite circuit
        associated with the extended commands.
    :type bipartite_circuit: BipartiteCircuit
    """
    # Add an extended command (EC) to a circuit,
    # with packing if needed as given by the supplied mvc
    # Cases are as follows
    # 1) The EC is 1q or not packable are no packings on the qubit.
    #   -> Just add it to circuit
    # 2) The EC is not packable but there are packings on the qubit
    #   which need to be terminated.
    #   -> Terminate all packings, add the EC to circuit
    # 4) The EC is packable as a 1q diagonal
    #   -> Just add it to circuit
    # 5) The EC is packable as a 1q antidiagonal
    #   -> Add to circuit and add correctional X gate to all current packings
    # 6) The EC is packable as a 2q CZ
    #   -> Add the CZ to the packing on the appropriate l_qubit.

    # Handle all the cases where the ExtendedCommand is local.
    if extended_command.is_local():
        vertex = extended_command.vertices[0]
        extended_qubit = vertex.extended_qubit
        # If not packable, we need to close off
        # all current packings on this qubit.
        if not extended_command.is_packable():
            while len(extended_qubit.current_l_qubits) > 0:
                l_qubit = extended_qubit.current_l_qubits[0]
                l_qubit.stop_packing()
                circuit.add_custom_gate(
                    end_proc,
                    [],
                    [
                        l_qubit.qubit,
                        l_qubit.origin_extended_qubit.qubit,
                    ],
                )

        elif extended_qubit.is_packing() and is_antidiagonal(
            extended_command.command.op
        ):
            for l_qubit in extended_qubit.current_l_qubits:
                circuit.X(l_qubit.qubit)

        circuit.add_gate(
            extended_command.command.op,
            extended_command.command.args,
        )
        vertex.added_extended_commands.append(extended_command)

    else:  # Handle cases of a global CZ
        if extended_command.command.op.type == OpType.CX:
            raise Exception("Global CX gates are not allowed.")
        for (
            vertex
        ) in extended_command.vertices:  # Add starting processes if needed
            if (
                vertex.vertex_index in bipartite_circuit.mvc
                and not vertex.is_packing
            ):
                le_register_index = vertex.get_connected_register_index()
                le_reg_name = (
                    f"Server {le_register_index}"
                    + f" Link Edge {bipartite_circuit.link_edge_count}"
                )
                bipartite_circuit.link_edge_count += 1
                le_q_reg = QubitRegister(le_reg_name, 1)
                circuit.add_q_register(le_q_reg)
                link_qubit = LinkQubit(le_q_reg[0], vertex)
                link_qubit.start_packing()
                circuit.add_custom_gate(
                    start_proc,
                    [],
                    [
                        link_qubit.origin_extended_qubit.qubit,
                        link_qubit.qubit,
                    ],
                )

        # If it's a global CZ, one of the two vertices it belongs to
        # must be in the MVC, so select that vertex.
        packed_vertex = [
            vertex
            for vertex in extended_command.vertices
            if vertex.vertex_index in bipartite_circuit.mvc
        ][0]
        l_qubit = packed_vertex.link_qubit
        qubit = [
            qubit
            for qubit in extended_command.command.qubits
            if qubit != l_qubit.origin_extended_qubit.qubit
        ][0]
        circuit.add_gate(
            extended_command.command.op,
            [qubit, l_qubit.qubit],
        )
        for vertex in extended_command.vertices:
            vertex.added_extended_commands.append(extended_command)

    for vertex in extended_command.vertices:
        # If all of the commands on the vertex are packed add an ending process
        if (
            vertex.vertex_index in bipartite_circuit.mvc
            and vertex.extended_commands == vertex.added_extended_commands
        ):
            l_qubit = vertex.link_qubit
            l_qubit.stop_packing()
            circuit.add_custom_gate(
                end_proc,
                [],
                [
                    l_qubit.qubit,
                    l_qubit.origin_extended_qubit.qubit,
                ],
            )


def add_edge_to_graph(vertex1, vertex2, graph):
    """Given two CommandVertexs and a graph,
    add an edge to the graph that connects the two vertices.
    If either of the vertices are not yet on the graph
    then also add them to the graph.

    :param vertex1: The first vertex we wish to add an edge between.
    :type vertex1: CommandVertex
    :param vertex2: The second vertex we wish to add an edge between.
    :type vertex2: CommandVertex
    :param graph: The graph to which the edge
        (and vertices) should be added to.
    :type graph: networkx.Graph
    """
    # Neither vertices are on the graph,
    # so can fix which half of the graph they are on.
    if not (vertex1.is_on_graph or vertex2.is_on_graph):
        vertex1.set_bipartite_i(0)
        vertex2.set_bipartite_i(1)
        graph.add_nodes_from(
            [vertex1.get_index()],
            bipartite=vertex1.get_bipartite_i(),
        )
        graph.add_nodes_from(
            [vertex2.get_index()],
            bipartite=vertex2.get_bipartite_i(),
        )

    elif vertex1.is_on_graph:
        vertex2.set_bipartite_i(
            [i for i in range(2) if i != vertex1.get_bipartite_i()][0]
        )
        graph.add_nodes_from(
            [vertex2.get_index()],
            bipartite=vertex2.get_bipartite_i(),
        )

    elif vertex2.is_on_graph:
        vertex1.set_bipartite_i(
            [i for i in range(2) if i != vertex2.get_bipartite_i()][0]
        )
        graph.add_nodes_from(
            [vertex1.get_index()],
            bipartite=vertex1.get_bipartite_i(),
        )

    graph.add_edges_from(
        [
            (
                vertex1.get_index(),
                vertex2.get_index(),
            )
        ]
    )


def to_extended_commands(commands):
    """Given a list of commands, convert them to ExtendedCommands.

    Typically this list of commands comes from
    pytket.circuit.Circuit.get_commands().
    Their index is given by the order in the list they occur at.

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

class ExtendedQubitRegister:
    def __init__(self, register, register_index):
        self.register = register
        self.name = register.name
        self.size = register.size
        self.register_index = register_index
        self.max_parallel_packings = 0
        self.current_packings = set()
        self.all_packings = set()

    def start_packing(self, from_qubit, to_register):
        self.current_packings.add({from_qubit, to_register})
        self.all_packings.add({from_qubit, to_register})
        if len(self.current_packings) > self.max_parallel_packings:
            self.max_parallel_packings = len(self.current_packings)

    def stop_packing(self, from_qubit, to_register):
        self.current_packings.remove({from_qubit, to_register})


class CommandVertex:
    def __init__(self, i, extended_qubit):
        self.vertex_index = i
        self.extended_qubit = extended_qubit
        self.register = extended_qubit.register
        self.extended_commands = []
        self.added_extended_commands = []
        self.is_open = False
        self.connected_register = None
        self.connected_vertices = set()
        self.is_on_graph = False
        self.is_packing = False
        self.link_qubit = None

    def close(self):
        self.is_open = False

    def get_connected_register_index(self):
        return self.connected_register.register_index

    def is_connected(self):
        return self.connected_register is not None

    def get_index(self):
        return self.vertex_index

    def get_register_index(self):
        return self.register.register_index

    def add_extended_command(self, extended_command):
        self.extended_commands.append(extended_command)

    def connect_to_register(self, connected_register):
        assert isinstance(connected_register, ExtendedQubitRegister), 'There is a bug with the input register'
        self.connected_register = connected_register
        self.is_open = True

    def connect_vertex(self, vertex):
        self.connected_vertices.add(vertex)
        assert vertex.get_register_index() == self.get_connected_register_index()
        self.is_open = True

    def get_extended_command_indices(self):
        indices = []
        for extended_command in self.extended_commands:
            indices.append(extended_command.command_index)
        return indices

    def set_bipartite_i(
        self, i
    ):  # The bipartite partition on which this vertex resides
        self.bipartite_i = i
        self.is_on_graph = True

    def get_bipartite_i(self):
        return self.bipartite_i


class ExtendedQubit:
    def __init__(self, qubit, register, extended_commands):
        self.qubit = qubit
        self.register = register
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

    def create_vertex(self, vertex_index):
        self.vertices.append(CommandVertex(vertex_index, self))
        self.last_used_vertex = self.vertices[-1]

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
        # A connected server is one from which this ExtendedQubit
        # has a StartingProcess leading to a Link Edge qubit on that server
        # A currently connected server is a connected server
        # for which the StartingProcess has not been ended via an EndingProcess
        # In essence this function finds every server
        # that this qubit is connected to
        # using the server number as keys to a list of vertices
        # that are connected to the relevant Link Edge qubit.

        # Maps the connected_server_num
        # -> all vertices connected to this vertex on that register
        connected_registers_dict = {}
        connected_vertices = self.get_currently_connected_vertices()
        for vertex in connected_vertices:
            if (
                vertex.get_connected_register_index()
                in connected_registers_dict.keys()
            ):
                connected_registers_dict[vertex.get_connected_register_index()].add(
                    vertex
                )
            else:
                connected_registers_dict[vertex.get_connected_register_index] = {
                    vertex
                }

        return connected_registers_dict

    def get_command_indices(self):
        command_indices = []
        for extended_command in self.get_extended_commands():
            command_indices.append(extended_command.get_index())
        return command_indices

    def set_last_used_vertex(self, vertex):
        self.last_used_vertex = vertex

    def close_all_vertices(self):
        for vertex in self.vertices:
            vertex.close()


class ExtendedCommand:
    def __init__(self, i, command):
        self.command_index = i
        self.command = command
        self.vertices = []
        self.extended_qubits = None

    def add_vertex(self, vertex):
        self.vertices.append(vertex)

    def get_vertices(self):
        return self.vertices

    def is_1q(self):
        return len(self.command.qubits) == 1

    def other_arg_qubit(self, this_extended_qubit):
        if self.extended_qubits is None:
            raise Exception(
                "This ExtendedCommand does not yet have"
                + " ExtendedQubits assigned to it."
            )
        return [
            extended_qubit
            for extended_qubit in self.extended_qubits
            if extended_qubit != this_extended_qubit
        ][0]

    def is_local(self):
        if self.is_1q():
            return True
        q0 = self.extended_qubits[0]
        return q0.register.register_index == self.other_arg_qubit(q0).register.register_index

    def get_index(self):
        return self.command_index

    def is_1q_packable(self):
        return self.is_1q() and self.is_packable()

    def is_packable(self):
        return is_diagonal(self.command.op) or is_antidiagonal(self.command.op)

    def set_extended_qubits(self, extended_qubits):
        self.extended_qubits = extended_qubits


class BipartiteCircuit:
    def __init__(
        self,
        circuit,
        placement=None,
        distributor=None,
        network=None,
    ):
        if not dqc_gateset_predicate.verify(circuit):
            raise Exception("The given circuit is not in the allowed gateset")
        self.circuit = circuit
        self.dist_circ = DistributedCircuit(circuit)
        if placement is not None:
            self.placement = placement
        else:
            if network is None:
                raise Exception(
                    "Missing optional argument network. "
                    + "(Since no placement has been specified, "
                    + "BipartiteCircuit requires a network "
                    + "to distribute the circuit over)."
                )
            if distributor is None:
                distributor = Random()
            self.placement = distributor.distribute(self.dist_circ, network)
        self.circuit = self.dist_circ.to_relabeled_registers(self.placement)
        (
            self.graph,
            self.extended_commands,
        ) = self.to_bipartite()
        self.link_edge_count = 0
        self.top_nodes = {
            n for n, d in self.graph.nodes(data=True) if d["bipartite"] == 0
        }
        self.matching = maximum_matching(self.graph, self.top_nodes)
        self.mvc = to_vertex_cover(
            self.graph,
            self.matching,
            self.top_nodes,
        )
        self.packed_circuit = pack_circuit(self)

    def get_graph(self):
        return self.graph

    def get_packed_circuit(self):
        return self.packed_circuit

    def get_ebit_cost(self):
        return len(self.mvc)

    def to_bipartite(self):
        """Given a circuit, create a bipartite graph representing that circuit.

        :return graph: The graph representation of the circuit.
        :rtype: networkx.Graph
        """
        extended_commands = to_extended_commands(
            self.circuit.get_commands()
        )  # Give each command an index to reference it by.
        next_vertex_index = 0
        graph = Graph()

        # Convert QubitRegisters to ExtendedQubitRegisters
        extended_qubit_registers = {}
        for i, q_register in enumerate(self.circuit.q_registers):
            extended_qubit_registers[i] = ExtendedQubitRegister(
                q_register,
                i
            )

        # Convert qubits to ExtendedQubits.
        # extended_qubits maps each qubit -> its ExtendedQubit
        extended_qubits = {}
        for i, qubit in enumerate(self.circuit.qubits):
            new_extended_qubit = ExtendedQubit(
                qubit,
                extended_qubit_registers[self.placement.placement[i]],
                [],
            )
            new_extended_qubit.create_vertex(next_vertex_index)
            extended_qubits[qubit] = new_extended_qubit
            next_vertex_index += 1

        # Populate the ExtendedCommands list for each ExtendedQubit
        for extended_command in extended_commands:
            extended_command_extended_qubits = []
            for qubit in extended_command.command.qubits:
                extended_command_extended_qubits.append(extended_qubits[qubit])
                extended_qubits[qubit].add_extended_command(extended_command)
            extended_command.set_extended_qubits(
                extended_command_extended_qubits
            )

        # Create all the vertices on each qubit
        for extended_qubit in extended_qubits.values():
            for extended_command in extended_qubit.get_extended_commands():
                if (
                    extended_command.is_packable()
                    and extended_command.is_local()
                ):  # Local packable gate
                    vertex = extended_qubit.last_used_vertex
                    extended_command.add_vertex(vertex)
                    vertex.add_extended_command(extended_command)

                elif (
                    extended_command.is_1q()
                ):  # 1 qubit non-(anti)diagonal gate
                    extended_qubit.close_all_vertices()
                    extended_qubit.create_vertex(next_vertex_index)
                    extended_command.add_vertex(
                        extended_qubit.last_used_vertex
                    )
                    extended_qubit.last_used_vertex.add_extended_command(
                        extended_command
                    )
                    next_vertex_index += 1

                else:  # Non-local CZ gate
                    currently_connected_servers = (
                        extended_qubit.get_currently_connected_servers()
                    )
                    if (
                        extended_command.other_arg_qubit(
                            extended_qubit
                        ).register.register_index
                        in currently_connected_servers.keys()
                    ):
                        # Currently connected from this qubit
                        # to the server of the other qubit for this CZ,
                        # so can add this CZ to this packing set.
                        vertex = list(
                            currently_connected_servers[
                                extended_command.other_arg_qubit(
                                    extended_qubit
                                ).server_num
                            ]
                        )[0]
                        extended_qubit.add_vertex(vertex)

                    else:
                        # Must create a new connection
                        # from this qubit to the server.
                        if extended_qubit.last_used_vertex.is_connected():
                            extended_qubit.create_vertex(next_vertex_index)
                            next_vertex_index += 1
                        else:
                            vertex = extended_qubit.last_used_vertex
                        other_register = extended_command.other_arg_qubit(
                            extended_qubit
                        ).register
                        vertex.connect_to_register(other_register)

                    extended_command.add_vertex(vertex)
                    vertex.add_extended_command(extended_command)

        # A list that will contain all the edges
        # that have been added to the graph as set(vertex1, vertex2)
        added_edges = []
        for extended_command in extended_commands:
            if not (
                extended_command.is_local()
                or set(extended_command.get_vertices()) in added_edges
            ):
                add_edge_to_graph(
                    extended_command.get_vertices()[0],
                    extended_command.get_vertices()[1],
                    graph,
                )
                added_edges.append(set(extended_command.get_vertices()))

        return graph, extended_commands
