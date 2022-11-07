"""
This module contains the BipartiteCircuit
class, which is used for analysing the ebit
cost of a distributed quantum circuit via
minimum vertex covers.
"""
from networkx import Graph  # type: ignore
from networkx.algorithms.bipartite import (  # type: ignore
    maximum_matching,
    to_vertex_cover,
)
from pytket import Circuit
from pytket.circuit import QubitRegister, OpType  # type: ignore
from pytket_dqc.circuits import HypergraphCircuit

from pytket_dqc.packing import (
    ExtendedCommand,
    ExtendedQubit,
    ExtendedRegister,
    LinkQubit,
)

from pytket_dqc.utils.gateset import (
    start_proc,
    end_proc,
    dqc_gateset_predicate,
)
from pytket_dqc.utils.op_analysis import is_antidiagonal


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
            raise Exception(
                "The inputted circuit is not in a valid gateset. " +
                "You can apply ``DQCPass`` from pytket_dqc.utils " +
                "on the circuit to rebase it to a valid gateset."
            )
        dist_circ = HypergraphCircuit(circuit)
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
        total_size_added = 0
        for i, q_register in enumerate(self.circuit.q_registers):
            extended_register = ExtendedRegister(i, q_register)
            self.extended_registers.append(extended_register)
            for qubit_index in range(
                total_size_added, total_size_added + q_register.size
            ):
                qubit = self.circuit.qubits[qubit_index]
                extended_qubit = ExtendedQubit(
                    qubit_index, qubit, extended_register
                )
                extended_qubit.create_vertex(qubit_index)
                extended_register.add_extended_qubit(extended_qubit)
                self.extended_qubits[qubit] = extended_qubit
            total_size_added += q_register.size

        # Commands -> ExtendedCommands
        # Also populate ExtendedQubits with ExtendedCommands
        for i, command in enumerate(self.circuit.get_commands()):
            extended_command = ExtendedCommand(i, command)
            for qubit in command.qubits:
                extended_qubit = self.extended_qubits[qubit]
                extended_qubit.extended_commands.append(extended_command)
                extended_command.extended_qubits.append(extended_qubit)
            self.extended_commands.append(extended_command)

    def build_vertices(self):
        """Builds the vertices on each ExtendedQubit
        to be used on the bipartite graph.
        """
        next_vertex_index = len(self.extended_qubits)
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
                    extended_qubit.create_vertex(next_vertex_index)
                    next_vertex_index += 1
                    vertex = extended_qubit.last_used_vertex

                # Gate is a non-local CU1 gate
                # => Add to relevant vertices
                else:
                    assert (
                        extended_command.get_op_type() == OpType.CU1
                    ), "Expected this to be a non-local CU1 gate."
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
                        assert (
                            len(vertex_candidates) == 1
                        ), "There is more than one candidate Vertex."
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
                            extended_qubit.create_vertex(next_vertex_index)
                            next_vertex_index += 1
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
        """Given an ExtendedCommand for a non-local CU1,
        add an edge between the two vertices on this ExtendedCommand

        :param extended_command: A (non-local CU1) ExtendedCommand
        :type extended_command: pytket_dqc.packing.ExtendedCommand
        """
        assert (
            extended_command.get_op_type() == OpType.CU1
        ), "This ExtendedCommand must be a CU1."
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
        2. EC is a non-local CU1 gate

        For 1. there are two subcases
        where adjustments to the circuit must be made
        before the gate can be added to the circuit:
        1. a) The gate is not packable
          -> Any ongoing packings must be ended.
        1. b) The gate is packable and antidiagonal
          -> For each ongoing packing,
             an X gate must be added to the link qubit.

        For 2. need to add the CU1 to the LinkQubit
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
                extended_command.get_op_type() == OpType.CU1
            ), "The command being added to the circuit should be a CU1"
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

            # Non-local CU1
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

        # If all non-local CU1s on this vertex
        # have been added to the circuit
        # then add an EndingProcess to the LinkQubits.
        # This prevents redundant correction X gates
        # from being added to the circuit.
        for vertex in extended_command.vertices:
            if (
                vertex.is_open
                and vertex.vertex_index in self.mvc
                and vertex.added_all_nonlocal_cu1s()
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
                vertex.close()

    def get_ebit_cost(self):
        """Computes the ebit cost of this
        distribution and packing of the circuit.

        :return: The ebit cost.
        :rtype: int
        """
        return len(self.mvc)
