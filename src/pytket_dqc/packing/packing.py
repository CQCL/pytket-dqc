"""
This module contains classes and methods
that utilise a minimum vertex cover approach
to finding the ebit cost of a distributed circuit.
"""
from pytket import Circuit
from pytket.circuit import QubitRegister, OpType  # type: ignore
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
from pytket_dqc.circuits import DistributedCircuit


class BipartiteCircuit:
    """Used for generating the bipartite representation of a circuit,
    and computing it's ebit cost in a setting
    where evicted gates are not allowed.
    """

    def __init__(self, circuit, placement):
        self.circuit = None
        self.placement = placement
        self.graph = Graph()
        self.matching = None
        self.mvc = None
        self.packed_circuit = Circuit()
        self.extended_commands = []
        self.extended_registers = []
        self.extended_qubits = {}
        self.next_vertex_index = 0
        self.top_vertices = None
        self.from_placed_circuit(circuit)

    def from_placed_circuit(self, circuit):
        """Generates a packed circuit from a placed circuit and its placement.

        :param circuit: The placed circuit.
        :type circuit: pytket.circuit.Circuit
        :param placement: The placement of the circuit.
        :type placement: pytket_dqc.placement.Placement
        :raises Exception: If the circuit is not in the allowed gateset.
        """
        if not dqc_gateset_predicate.verify(circuit):
            raise Exception("The given circuit is not in the allowed gateset.")
        dist_circ = DistributedCircuit(circuit)
        self.circuit = dist_circ.to_relabeled_registers(self.placement)
        self.build_bipartite_graph()
        self.build_top_vertices()
        self.find_matching()
        self.find_mvc()
        self.pack_circuit()

    def build_top_vertices(self):
        """Creates a set of the vertices
        in the 'top half' of the bipartite graph.
        """
        self.top_vertices = {
            n for n, d in self.graph.nodes(data=True) if d["bipartite"] == 0
        }

    def find_matching(self):
        """Finds a matching, using the already built set of top vertices."""
        self.matching = maximum_matching(self.graph, self.top_vertices)

    def find_mvc(self):
        """Finds a minimum vertex cover of the prebuilt
        bipartite graph representation of the circuit.
        """
        self.mvc = to_vertex_cover(
            self.graph, self.matching, self.top_vertices
        )

    def convert_to_extended(self):
        """Converts various pytket classes to their extended counterparts."""
        # QubitRegisters -> ExtendedRegisters
        # Qubits -> ExtendedQubits
        # Populate ExtendedRegisters with ExtendedQubits
        qubit_index = 0
        total_size_added = 0
        for i, q_register in enumerate(self.circuit.q_registers):
            extended_register = ExtendedRegister(i, q_register)
            self.extended_registers.append(extended_register)
            while qubit_index - total_size_added < q_register.size:
                qubit = self.circuit.qubits[qubit_index]
                extended_qubit = ExtendedQubit(
                    qubit_index, qubit, extended_register
                )
                extended_qubit.create_vertex(self.next_vertex_index)
                self.next_vertex_index += 1
                extended_register.add_extended_qubit(extended_qubit)
                self.extended_qubits[qubit] = extended_qubit
                qubit_index += 1
            total_size_added += q_register.size

        # Commands -> ExtendedCommands
        # Also populate ExtendedQubits with ExtendedCommands
        for i, command in enumerate(self.circuit.get_commands()):
            extended_command = ExtendedCommand(i, command)
            for qubit in command.qubits:
                extended_qubit = self.extended_qubits[qubit]
                extended_qubit.add_extended_command(extended_command)
                extended_command.extended_qubits.append(extended_qubit)
            self.extended_commands.append(extended_command)

    def build_vertices(self):
        """Builds the vertices on each ExtendedQubit
        to be used on the bipartite graph.
        """
        for extended_qubit in self.extended_qubits.values():
            for extended_command in extended_qubit.extended_commands:
                # Gate is packable and local
                # => The vertex to add this gate to can
                # just be the last vertex on the ExtendedQubit
                if (
                    extended_command.is_packable()
                    and extended_command.is_local()
                ):
                    vertex = extended_qubit.last_used_vertex

                # Gate is some other (single) qubit gate
                # => Must end all vertices on this
                # ExtendedQubit and start a new one
                elif extended_command.is_1q():
                    extended_qubit.close_all_vertices()
                    extended_qubit.create_vertex(self.next_vertex_index)
                    self.next_vertex_index += 1
                    vertex = extended_qubit.last_used_vertex

                # Gate is a non-local CZ gate
                # => Add to relevant vertices
                else:
                    assert (
                        extended_command.get_op_type() == OpType.CZ
                    ), "Expected this to be a non-local CZ gate."
                    currently_linked_registers = (
                        extended_qubit.get_currently_linked_registers()
                    )
                    other_qubit = extended_command.other_arg_qubit(
                        extended_qubit
                    )

                    # There is already a vertex that
                    # has been linked to another register
                    # => Set the vertex equal to that one
                    if other_qubit.register in currently_linked_registers:
                        vertex_candidates = [
                            vertex
                            for vertex in extended_qubit.get_in_use_vertices()
                            if vertex.linked_register == other_qubit.register
                        ]
                        assert len(vertex_candidates) == 1, \
                            "There is more than one candidate Vertex."
                        # Pick out the vertex which is
                        # linked to the correct register
                        vertex = vertex_candidates[0]

                    # There is no vertex from this
                    # register linked to the other register
                    # => Make it
                    else:
                        # The last used vertex is linked
                        # => All vertices on this ExtendedQubit are
                        # linked to different registers from the desired one
                        # => Make a new vertex and link it to
                        # the relevant register
                        if extended_qubit.last_used_vertex.is_linked:
                            extended_qubit.create_vertex(
                                self.next_vertex_index
                            )
                            self.next_vertex_index += 1
                        vertex = extended_qubit.last_used_vertex
                        vertex.link_to_register(other_qubit.register)

                extended_command.add_vertex(vertex)
                vertex.add_extended_command(extended_command)

    def build_bipartite_graph(self):
        """Builds the bipartite graph representing the given circuit."""
        # Convert Commands, QubitRegisters,
        # Qubits to their Extended counterparts
        self.convert_to_extended()

        # Build vertices on qubits
        self.build_vertices()

        # Add edges to the graph (implicitly adds vertices)
        added_edges = (
            []
        )  # Contains the edges that have been added to the graph already
        for extended_command in self.extended_commands:
            if not (
                extended_command.is_local()
                or set(extended_command.vertices) in added_edges
            ):
                self.add_edge_to_graph(extended_command)
                added_edges.append(set(extended_command.vertices))

    def add_edge_to_graph(self, extended_command):
        """Given an ExtendedCommand for a non-local CZ,
        add an edge between the two vertices on this ExtendedCommand

        :param extended_command: A (non-local CZ) ExtendedCommand
        :type extended_command: pytket_dqc.packing.ExtendedCommand
        """
        assert (
            extended_command.get_op_type() == OpType.CZ
        ), "This ExtendedCommand must be a CZ."
        assert (
            not extended_command.is_local()
        ), "This ExtendedCommand must be non-local."
        vertex0 = extended_command.vertices[0]
        vertex1 = extended_command.vertices[1]

        # Neither vertex is on graph
        # -> Choose arbitrarily which half of graph
        if not (vertex0.is_on_graph or vertex1.is_on_graph):
            vertex0.set_is_top_half(True)
            vertex1.set_is_top_half(False)
            self.graph.add_nodes_from(
                [vertex0.vertex_index], bipartite=vertex0.is_top_half
            )
            self.graph.add_nodes_from(
                [vertex1.vertex_index], bipartite=vertex1.is_top_half
            )

        # If either vertex is not yet on the graph, then add to graph
        elif vertex0.is_on_graph:
            vertex1.set_is_top_half(not vertex0.is_top_half)
            self.graph.add_nodes_from(
                [vertex1.vertex_index], bipartite=vertex1.is_top_half
            )

        elif vertex1.is_on_graph:
            vertex0.set_is_top_half(not vertex1.is_top_half)
            self.graph.add_nodes_from(
                [vertex0.vertex_index], bipartite=vertex0.is_top_half
            )

        # Add edge between the two vertices
        self.graph.add_edges_from(
            [(vertex0.vertex_index, vertex1.vertex_index)]
        )

    def pack_circuit(self):
        """Creates the circuit with Starting and EndingProcesses."""
        for qubit in self.circuit.qubits:
            self.packed_circuit.add_qubit(qubit)

        for extended_command in self.extended_commands:
            self.add_extended_command_to_packed_circuit(extended_command)

    def add_extended_command_to_packed_circuit(self, extended_command):
        """Adds a given extended command to the circuit,
        as well as Starting and EndingProcesses if needed.

        Cases are divided into two types:
        1. EC is local gate
        2. EC is a non-local CZ gate

        For 1. there are two subcases
        where adjustments to the circuit must be made
        before the gate can be added to the circuit:
        1. a) The gate is not packable
          -> Any ongoing packings must be ended.
        1. b) The gate is packable and antidiagonal
          -> For each ongoing packing,
             an X gate must be added to the link qubit.

        For 2. need to add the CZ to the LinkQubit
        that one of the argument qubits has been teleported to.
        If such a LinkQubit does not yet exist
        then it needs to be created.

        :param extended_command: The ExtendedCommand to add to the circuit
        :type extended_command: pytket_dqc.packing.ExtendedCommand
        """

        circuit = self.packed_circuit

        # Case 1.
        if extended_command.is_local():
            vertex = extended_command.vertices[0]
            extended_qubit = vertex.extended_qubit

            # Case 1. a)
            if not extended_command.is_packable():
                while len(extended_qubit.in_use_link_qubits) > 0:
                    link_qubit = extended_qubit.in_use_link_qubits[0]
                    link_qubit.end_packing()
                    circuit.add_custom_gate(
                        end_proc,
                        [],
                        [
                            link_qubit.qubit,
                            link_qubit.get_origin_extended_qubit().qubit,
                        ],
                    )

            # Case 1. b)
            elif extended_qubit.is_packing and is_antidiagonal(
                extended_command.command.op
            ):
                for link_qubit in extended_qubit.in_use_link_qubits:
                    circuit.X(link_qubit.qubit)

            circuit.add_gate(
                extended_command.command.op, extended_command.command.args
            )
            vertex.added_extended_commands.append(extended_command)

        # Case 2.
        else:
            assert (
                extended_command.get_op_type() == OpType.CZ
            ), "The command being added to the circuit should be a CZ"
            for vertex in extended_command.vertices:
                # Make a LinkQubit if one does not already exist
                # and add a starting process to it
                if vertex.vertex_index in self.mvc and not vertex.is_packing:
                    assert (
                        vertex.is_linked
                    ), f"There is no linked register on vertex \
                        {vertex.vertex_index}"
                    link_register = vertex.linked_register
                    link_register_name = (
                        f"{link_register.get_name()}"
                        + f" Link Edge {link_register.link_count}"
                    )
                    link_register.link_count += 1
                    link_fake_register = QubitRegister(
                        link_register_name, 1
                    )  # The qubit cannot be directly added to the circuit
                    # so we add it by adding a register
                    # and adding the qubit to that
                    circuit.add_q_register(link_fake_register)
                    link_qubit = LinkQubit(link_fake_register[0], vertex)
                    link_qubit.start_packing()
                    circuit.add_custom_gate(
                        start_proc,
                        [],
                        [
                            link_qubit.get_origin_extended_qubit().qubit,
                            link_qubit.qubit,
                        ],
                    )

            # Non-local CZ
            # -> one of it's two vertices must be in self.mvc.
            # Select that vertex.
            packed_vertex = [
                vertex
                for vertex in extended_command.vertices
                if vertex.vertex_index in self.mvc
            ][0]
            link_qubit = packed_vertex.link_qubit
            unteleported_qubit = [
                qubit
                for qubit in extended_command.command.qubits
                if qubit != link_qubit.get_origin_extended_qubit().qubit
            ][0]

            circuit.add_gate(
                extended_command.command.op,
                [unteleported_qubit, link_qubit.qubit],
            )

            for vertex in extended_command.vertices:
                vertex.added_extended_commands.append(extended_command)

        # Now check if all the ExtendedCommands on a vertex have been added
        # If so, then add an EndingProcess to the LinkQubits
        for vertex in extended_command.vertices:
            if (
                vertex.vertex_index in self.mvc
                and vertex.extended_commands == vertex.added_extended_commands
            ):
                link_qubit = vertex.link_qubit
                link_qubit.end_packing()
                circuit.add_custom_gate(
                    end_proc,
                    [],
                    [
                        link_qubit.qubit,
                        link_qubit.get_origin_extended_qubit().qubit,
                    ],
                )

    def get_ebit_cost(self):
        """Computes the ebit cost of this
        distribution and packing of the circuit.

        :return: The ebit cost.
        :rtype: int
        """
        return len(self.mvc)


class ExtendedRegister:
    """Extends the pytket.circuit.QubitRegister class."""

    def __init__(self, register_index, register):
        self.register = register
        self.register_index = register_index
        self.max_parallel_packings = 0
        self.current_packings = set()
        self.all_packings = set()
        self.link_count = 0
        self.extended_qubits = []

    def add_extended_qubit(self, extended_qubit):
        """Adds an ExtendedQubit to this ExtendedRegister.

        :param extended_qubit: The ExtendedQubit to add.
        :type extended_qubit: pytket_dqc.packing.ExtendedQubit
        """
        self.extended_qubits.append(extended_qubit)

    def get_name(self):
        """Returns the name of the underlying register.

        :return: The name of the register.
        :rtype: str
        """
        return self.register.name


class ExtendedQubit:
    """Extends the pytket.circuit.Qubit class."""

    def __init__(self, qubit_index, qubit, register):
        self.qubit_index = qubit_index
        self.qubit = qubit
        self.register = register
        self.extended_commands = []
        self.vertices = []
        self.last_used_vertex = None
        self.all_link_qubits = []
        self.in_use_link_qubits = []

    def is_packing(self):
        """Checks to see if there are any ongoing packings on this qubit.

        :return: If there are any ongoing packings.
        :rtype: bool
        """
        return len(self.in_use_link_qubits) > 0

    def start_link_to_qubit(self, link_qubit):
        """Begins a link to a supplied LinkQubit.

        :param link_qubit: The LinkQubit to link to.
        :type link_qubit: pytket_dqc.packing.LinkQubit
        """
        self.all_link_qubits.append(link_qubit)
        self.in_use_link_qubits.append(link_qubit)

    def end_link_to_qubit(self, link_qubit):
        """Ends a link to a given LinkQubit.

        :param link_qubit: The LinkQubit to end the link to.
        :type link_qubit: pytket_dqc.packing.LinkQubit
        """
        assert (
            link_qubit in self.in_use_link_qubits
        ), "Cannot end a link to a qubit which is not currently linked to. "
        self.in_use_link_qubits.remove(link_qubit)

    def add_extended_command(self, extended_command):
        """Adds an ExtendedCommand to this circuit.

        :param extended_command: The ExtendedCommand to add.
        :type extended_command: ExtendedCommand
        """
        self.extended_commands.append(extended_command)

    def create_vertex(self, vertex_index):
        """Creates a new vertex on this ExtendedQubit.

        :param vertex_index: The index of this vertex.
        :type vertex_index: int
        """
        self.vertices.append(Vertex(vertex_index, self))
        self.last_used_vertex = self.vertices[-1]

    def add_vertex(self, vertex):
        """Adds a vertex to this ExtendedQubit.

        :param vertex: The vertex to add.
        :type vertex: Vertex
        """
        self.vertices.append(vertex)
        self.last_used_vertex = self.vertices[-1]

    def get_in_use_vertices(self):
        """Finds all the vertices currently in use.

        :return: A set of in use vertices.
        :rtype: set(Vertex)
        """
        in_use_vertices = set()
        for vertex in self.vertices:
            if vertex.is_open and vertex.is_linked:
                in_use_vertices.add(vertex)
        return in_use_vertices

    def get_currently_linked_registers(self):
        """Find all the registers currently linked to this ExtendedQubit.

        :return: Dictionary of all the registers linked to this ExtendedQubit.
        :rtype: dict{ExtendedRegister: Vertex}
        """
        linked_registers_dict = {}
        linked_vertices = self.get_in_use_vertices()
        for vertex in linked_vertices:
            if vertex.linked_register in linked_registers_dict.keys():
                linked_registers_dict[vertex.linked_register].add(vertex)
            else:
                linked_registers_dict[vertex.linked_register] = {vertex}

        return linked_registers_dict

    def close_all_vertices(self):
        """Close every vertex on this ExtendedQubit."""
        for vertex in self.vertices:
            vertex.close()


class ExtendedCommand:
    """Extends the pytket.circuit.Command class."""

    def __init__(self, command_index, command):
        self.command_index = command_index
        self.command = command
        self.vertices = []
        self.extended_qubits = []

    def add_vertex(self, vertex):
        """Adds a Vertex to this ExtendedCommand

        :param vertex: The Vertex to add.
        :type vertex: pytket_dqc.packing.Vertex
        """
        self.vertices.append(vertex)

    def is_1q(self):
        """Checks if this ExtendedCommand is 1 qubit.

        :return: If this ExtendedCommand is 1 qubit.
        :rtype: bool
        """
        return len(self.command.qubits) == 1

    def other_arg_qubit(self, arg_extended_qubit):
        """Given one argument ExtendedQubit, return the other one.

        :param arg_extended_qubit: The known ExtendedQubit
        :type arg_extended_qubit: pytket.packing.ExtendedQubit
        :return: The other arguement ExtendedQubit.
        :rtype: pytket.packing.ExtendedQubit
        """
        assert (
            len(self.extended_qubits) == 2
        ), "This ExtendedCommand does not have \
        two ExtendedQubits associated with it."

        assert arg_extended_qubit in self.extended_qubits,\
            "The given argument ExtendedQubit is not\
            an argument ExtendedQubit on this\
            ExtendedCommand."

        assert len([
            extended_qubit
            for extended_qubit in self.extended_qubits
            if extended_qubit != arg_extended_qubit
        ]) == 1, "There are multiple other arguement\
            ExtendedQubits."

        return [
            extended_qubit
            for extended_qubit in self.extended_qubits
            if extended_qubit != arg_extended_qubit
        ][0]

    def is_local(self):
        """Checks if this is a local ExtendedCommand.

        :return: If the ExtendedCommand is local.
        :rtype: bool
        """
        assert (
            len(self.extended_qubits) > 0
        ), "This ExtendedCommand has no ExtendedQubits assigned to it."
        if self.is_1q():
            return True
        q_0 = self.extended_qubits[0]
        return (
            q_0.register.register_index
            == self.other_arg_qubit(q_0).register.register_index
        )

    def is_1q_packable(self):
        """Checks if this ExtendedCommand is both packable and single qubit.

        :return: If the ExtendedCommand is both packable and single qubit.
        :rtype: bool
        """
        return self.is_1q() and self.is_packable()

    def is_packable(self):
        """Checks if the ExtendedCommand is packable.

        :return: If the ExtendedCommand is packable.
        :rtype: bool
        """
        return is_diagonal(self.command.op) or is_antidiagonal(self.command.op)

    def get_op_type(self):
        """Finds the OpType of the underlying command.

        :return: The OpType of the underlying command.
        :rtype: pytket.circuit.OpType
        """
        return self.command.op.type


class LinkQubit:
    """A class representing qubits that are link qubits on the circuit.

    In essence, these are entangled ancilla qubits.
    """

    def __init__(self, qubit, vertex):
        self.qubit = qubit
        self.vertex = vertex
        self.is_open = False

    def get_origin_extended_qubit(self):
        """Finds the underlying original qubit that this qubit links to.

        :return: The original ExtendedQubit this LinkQubit is linked to.
        :rtype: pytket_dqc.packing.ExtendedQubit.
        """
        return self.vertex.extended_qubit

    def start_packing(self):
        """Begin a packing on this LinkQubit."""
        assert (
            not self.is_open
        ), "A packing on this LinkQubit has already begun."
        self.is_open = True
        self.vertex.is_packing = True
        self.vertex.link_qubit = self
        self.get_origin_extended_qubit().start_link_to_qubit(self)

    def end_packing(self):
        """End the packing on this LinkQubit."""
        assert self.is_open, "There's no packing on this LinkQubit to end."
        self.is_open = False
        self.vertex.is_packing = False
        self.get_origin_extended_qubit().end_link_to_qubit(self)


class Vertex:
    """Represents the vertices of the
    bipartite graph representation of the circuit.
    """

    def __init__(self, vertex_index, extended_qubit):
        self.vertex_index = vertex_index
        self.extended_qubit = extended_qubit
        self.extended_commands = []
        self.added_extended_commands = []
        self.is_open = False
        self.linked_register = None
        self.linked_vertices = set()
        self.is_on_graph = False
        self.is_packing = False
        self.link_qubit = None
        self.is_top_half = None
        self.is_linked = False

    def get_register(self):
        """Finds the ExtendedRegister that this Vertex lies on.

        :return: The ExtendedRegister this vertex lies on.
        :rtype: pytket_dqc.packing.ExtendedRegister
        """
        return self.extended_qubit.register

    def close(self):
        """Marks that this Vertex is no longer packing."""
        self.is_open = False

    def add_extended_command(self, extended_command):
        """Adds an ExtendedCommand to this Vertex.

        :param extended_command: The ExtendedCommand to add.
        :type extended_command: pytket_dqc.packing.ExtendedCommand.
        """
        self.extended_commands.append(extended_command)

    def link_to_register(self, register_to_link):
        """Link this vertex to a given register.

        :param register_to_link: The register to link to.
        :type register_to_link: pytket_dqc.packing.ExtendedRegister
        """
        assert isinstance(
            register_to_link, ExtendedRegister
        ), "Was not given a register to link to."
        self.linked_register = register_to_link
        self.is_open = True
        self.is_linked = True

    def get_extended_command_indices(self):
        """Find the indices of the ExtendedCommands in this Vertex.

        :return: A list of the indices of the ExtendedCommands in this Vertex.
        :rtype: List[int]
        """
        indices = []
        for extended_command in self.extended_commands:
            indices.append(extended_command.command_index)
        return indices

    def set_is_top_half(self, is_top_half):
        """Set whether this Vertex is in the
        'top half' of the bipartite graph or not.

        :param is_top_half: Is this Vertex in the top half.
        :type is_top_half: bool
        """
        self.is_top_half = is_top_half
        self.is_on_graph = True
