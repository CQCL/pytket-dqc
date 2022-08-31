"""
This module contains classes and methods
that utilise a minimum vertex cover approach
to finding the ebit cost of a distributed circuit.
"""

from pytket.circuit import OpType  # type: ignore

from pytket_dqc.utils.op_analysis import (
    is_diagonal,
    is_antidiagonal,
)


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

        assert (
            arg_extended_qubit in self.extended_qubits
        ), "The given argument ExtendedQubit is not\
            an argument ExtendedQubit on this\
            ExtendedCommand."

        assert (
            len(
                [
                    extended_qubit
                    for extended_qubit in self.extended_qubits
                    if extended_qubit != arg_extended_qubit
                ]
            )
            == 1
        ), "There are multiple other arguement\
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

    def added_all_nonlocal_czs(self):
        """Check if all the non-local CRzs on this
        circuit have been added to the circuit.


        :return: A bool describing if all the
        non-local CRzs have been added.
        :rtype: bool
        """
        nonlocal_crz_count = 0
        nonlocal_added_crz_count = 0

        for extended_command in self.extended_commands:
            if (
                extended_command.get_op_type() == OpType.CRz
                and not extended_command.is_local()
            ):
                nonlocal_crz_count += 1

        for extended_command in self.added_extended_commands:
            if (
                extended_command.get_op_type() == OpType.CRz
                and not extended_command.is_local()
            ):
                nonlocal_added_cz_count += 1

        return nonlocal_crz_count == nonlocal_added_crz_count
