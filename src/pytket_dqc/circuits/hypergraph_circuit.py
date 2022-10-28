from __future__ import annotations

from .hypergraph import Hypergraph, Hyperedge, Vertex
from pytket import OpType, Circuit, Qubit
from pytket.circuit import Command, Unitary2qBox  # type: ignore
from scipy.stats import unitary_group  # type: ignore
import numpy as np
from pytket.passes import DecomposeBoxes  # type: ignore
import networkx as nx  # type: ignore
import random
from pytket_dqc.utils import (
    dqc_gateset_predicate,
    DQCPass,
)
from pytket_dqc.utils.gateset import start_proc, end_proc, telep_proc

from typing import TYPE_CHECKING, cast, Union

if TYPE_CHECKING:
    from pytket.circuit import QubitRegister  # type: ignore
    from pytket_dqc import Placement, NISQNetwork


class HypergraphCircuit(Hypergraph):
    """Class representing circuit to be distributed on a network.
    HypergraphCircuit is a child of Hypergraph. HypergraphCircuit adds
    additional information on top of Hypergraph which describes the
    correspondence to a circuit.

    :param _circuit: Circuit to be distributed.
    :type _circuit: Circuit
    :param _vertex_circuit_map: Map from hypergraph vertices to circuit
        commands.
    :type _vertex_circuit_map: dict[int, dict]
    """

    def __init__(self, circuit: Circuit):
        """ Initialisation function

        :param circuit: Circuit to be distributed.
        :type circuit: Circuit
        """

        self.reset(circuit)
        assert self._vertex_id_predicate()

    def __str__(self):
        out_string = super().__str__()
        out_string += f"\nVertex Circuit Map: {self._vertex_circuit_map}"
        out_string += "\nCircuit: " + self._circuit.__str__()
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

        self._circuit = circuit
        self._vertex_circuit_map: dict[int, dict] = {}
        self._commands: list[
            dict[str, Union[Command, str, int, list[Qubit]]]
        ] = []
        self.from_circuit()

    def get_circuit(self):
        return self._circuit.copy()

    def add_hyperedge(
        self,
        vertices: list[Vertex],
        weight: int = 1,
        hyperedge_list_index: int = None,
        hyperedge_dict_index: list[int] = None
    ):
        """Add hyperedge to hypergraph of circuit. Adds some checks on top
        of add_hypergraph in `Hypergraph` in order to ensure that the
        first vertex is a qubit, and that there is only one qubit vertex.

        :param vertices: List of vertices in hyperedge to add.
        :type vertices: list[Vertex]
        :param weight: Hyperedge weight
        :type weight: int
        :param hyperedge_list_index: index in `hyperedge_list` at which the new
            hyperedge will be added.
        :type hyperedge_list_index: int
        :param hyperedge_dict_index: index in `hyperedge_dict` at which the new
            hyperedge will be added. Note that `hyperedge_dict_index` should
            be the same length as vertices.
        :raises Exception: Raised if first vertex in hyperedge is not a qubit
        :raised Exception: Raised if there is more than one qubit
            vertex in the list of vertices.
        """

        if not self.is_qubit_vertex(vertices[0]):
            raise Exception(
                f"The first element of {vertices} " +
                "is required to be a qubit vertex."
            )

        if any(self.is_qubit_vertex(vertex) for vertex in vertices[1:]):
            raise Exception("There must be only one qubit in the hyperedge.")

        # I believe this is assumed when distribution costs are calculated.
        # Please correct me if this is unnecessary.
        if any(
            vertex >= next_vertex
            for vertex, next_vertex in zip(vertices[:-1], vertices[1:])
        ):
            raise Exception("Vertex indices must be in increasing order.")

        super(HypergraphCircuit, self).add_hyperedge(
            vertices=vertices,
            weight=weight,
            hyperedge_list_index=hyperedge_list_index,
            hyperedge_dict_index=hyperedge_dict_index
        )

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
        self._vertex_circuit_map[vertex] = {"type": "qubit", "node": qubit}

    def get_qubit_vertices(self) -> list[int]:
        """Return list of vertices which correspond to qubits

        :return: list of vertices which correspond to qubits
        :rtype: List[int]
        """
        return [
            vertex
            for vertex in self.vertex_list
            if self._vertex_circuit_map[vertex]["type"] == "qubit"
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
        self._vertex_circuit_map[vertex] = {"type": "gate", "command": command}

    def is_qubit_vertex(self, vertex: int) -> bool:
        """Checks if the given vertex corresponds to a qubit.

        :param vertex: Vertex to be checked.
        :type vertex: int
        :return: Is it a qubit vertex.
        :rtype: bool
        """
        return self._vertex_circuit_map[vertex]["type"] == "qubit"

    def get_qubit_vertex(self, hyperedge: Hyperedge) -> int:
        """Returns the qubit vertex in ``hyperedge``.
        """
        qubit_list = [
            vertex
            for vertex in hyperedge.vertices
            if self.is_qubit_vertex(vertex)
        ]

        assert len(qubit_list) == 1
        return qubit_list[0]

    def get_vertex_of_qubit(self, qubit: Qubit) -> int:
        """Returns the vertex that corresponds to ``qubit``.
        """
        vertex_list = [
            vertex
            for vertex in self.vertex_list
            if self.is_qubit_vertex(vertex)
            if self._vertex_circuit_map[vertex]['node'] == qubit
        ]

        assert len(vertex_list) == 1
        return vertex_list[0]

    def get_gate_vertices(self, hyperedge: Hyperedge) -> list[int]:
        """Returns the list of gate vertices in ``hyperedge``.
        """
        gate_vertex_list = [
            vertex
            for vertex in hyperedge.vertices
            if self._vertex_circuit_map[vertex]["type"] == "gate"
        ]

        assert len(gate_vertex_list) == len(hyperedge.vertices) - 1
        return gate_vertex_list

    def get_hyperedge_subcircuit(self, hyperedge: Hyperedge) -> list[Command]:
        """Returns the list of commands between the first and last gate within
        the hyperedge. Commands that don't act on the qubit vertex are omitted
        but embedded gates within the hyperedge are included.
        """
        hyp_qubit = self._vertex_circuit_map[self.get_qubit_vertex(hyperedge)][
            "node"
        ]
        gate_vertices = self.get_gate_vertices(hyperedge)
        if not gate_vertices:
            return []
        circ_commands = self._circuit.get_commands()

        # We will abuse the fact that, by construction, the gate vertices are
        # numbered from smaller to larger integers as we read the circuit from
        # left to right.
        # A solution that didn't use the trick would be preferable, but that'd
        # require changing ``vertex_circuit_map`` to point at indices in the
        # circuit.get_commands() list. Unfortunately, comparison of bindings
        # via the "is" keyword does not work here because "get_commands"
        # returns a deep copy of the command list (on each call, the Command
        # objects are different).
        subcirc_commands = []
        first_gate = min(gate_vertices)
        last_gate = max(gate_vertices)
        current_vertex_id = len(self._circuit.qubits)

        first_found = False
        for cmd in circ_commands:
            if current_vertex_id == first_gate and cmd.op.type == OpType.CU1:
                first_found = True
            if first_found and hyp_qubit in cmd.qubits:
                subcirc_commands.append(cmd)
            if current_vertex_id == last_gate and cmd.op.type == OpType.CU1:
                break
            if cmd.op.type == OpType.CU1:
                current_vertex_id += 1

        assert first_found and current_vertex_id == last_gate
        assert subcirc_commands[0].op.type == OpType.CU1
        assert subcirc_commands[-1].op.type == OpType.CU1

        return subcirc_commands

    def h_embedding_required(self, hyperedge: Hyperedge) -> bool:
        """Returns whether or not H-type embedding of CU1 gates is required
        to implement the given hyperedge.
        """
        commands = self.get_hyperedge_subcircuit(hyperedge)

        currently_embedding = False
        for cmd in commands:

            if cmd.op.type == OpType.H:
                currently_embedding = not currently_embedding
            elif currently_embedding and cmd.op.type == OpType.CU1:
                return True

        return False

    def from_circuit(self):
        """Method to create a hypergraph from a circuit.

        :raises Exception: Raised if the circuit whose hypergraph is to be
        created is not in the valid gate set.
        """

        if not dqc_gateset_predicate.verify(self._circuit):
            raise Exception("The inputted circuit is not in a valid gateset.")

        two_q_gate_count = 0
        # For each command in the circuit, add the command to a list.
        # If the command is a two-qubit gate, store n, where the
        # command is the nth 2 qubit gate in the circuit.
        for command in self._circuit.get_commands():
            if command.op.type in [OpType.CZ, OpType.CU1, OpType.CX]:
                self._commands.append(
                    {"command": command, "two q gate count": two_q_gate_count}
                )
                two_q_gate_count += 1
            else:
                self._commands.append({"command": command})

        # Construct the hypergraph corresponding to this circuit.
        # For each qubit, add commands acting on the qubit in an uninterrupted
        # sequence (i.e. not separated by single qubit gates) to the
        # same hyperedge, along with the qubit. If the gate is a CX,
        # add the gate vertex when the control is intercepted, and add a new
        # weight 2 hyper edge if the control is intercepted. The weight 2
        # hyperedge corresponds to performing a possible teleportation.
        for qubit_index, qubit in enumerate(self._circuit.qubits):

            self.add_qubit_vertex(qubit_index, qubit)

            hyperedge = [qubit_index]
            # Gather all of the commands acting on qubit.
            qubit_commands = [
                {"command_index": command_index, "command": command}
                for command_index, command in enumerate(self._commands)
                if qubit in command["command"].qubits
            ]

            # This tracks if any two qubit gates have been found on the qubit
            # wire. If not, a one element hyperedge containing just the
            # qubit vertex is added.
            two_qubit_gate_found = False

            for command_dict in qubit_commands:
                command = command_dict["command"]
                command_index = command_dict["command_index"]
                # If the command is a CZ gate add it to the current working
                # hyperedge.
                if command["command"].op.type in [OpType.CZ, OpType.CU1]:
                    two_qubit_gate_found = True
                    vertex = (
                        command["two q gate count"] + self._circuit.n_qubits
                    )
                    self.add_gate_vertex(vertex, command["command"])
                    hyperedge.append(vertex)
                    self._commands[command_index]["vertex"] = vertex
                    self._commands[command_index]["type"] = "distributed gate"
                # If the command is a CX, add it to the current
                # working hyperedge, if the working qubit is the control.
                # Otherwise start a fresh weight 2 hyper edge, add the two
                # vertex hyperedge consisting of the gate and the qubit, and
                # start a fresh hyper edge again.
                # TODO: Note that this method of adding a CX is very
                # lazy. Indeed, in the case where a teleportation is required,
                # a new hyper edge need not be started, as other gates which
                # follow may also benefit from the teleportation.
                elif command["command"].op.type in [OpType.CX]:
                    two_qubit_gate_found = True
                    # Check if working qubit is the control
                    if qubit == command["command"].qubits[0]:
                        vertex = (
                            command["two q gate count"]
                            + self._circuit.n_qubits
                        )
                        self.add_gate_vertex(vertex, command["command"])
                        hyperedge.append(vertex)
                    else:
                        # Add current working hyperedge.
                        if len(hyperedge) > 1:
                            self.add_hyperedge(hyperedge)
                        # Add two vertex weight 2 hyperedge
                        vertex = (
                            command["two q gate count"]
                            + self._circuit.n_qubits
                        )
                        self.add_gate_vertex(vertex, command["command"])
                        hyperedge = [qubit_index, vertex]
                        self.add_hyperedge(hyperedge, weight=2)
                        # Start a fresh hyperedge
                        hyperedge = [qubit_index]
                    self._commands[command_index]["vertex"] = vertex
                    self._commands[command_index]["type"] = "distributed gate"
                # Otherwise (which is to say if a single qubit gate is
                # encountered) add the hyperedge to the hypergraph and start
                # a new working hyperedge.
                else:
                    if len(hyperedge) > 1:
                        self.add_hyperedge(hyperedge)
                        hyperedge = [qubit_index]
                    self._commands[command_index]["type"] = "1q local gate"

            # If there is an hyperedge that has not been added once all
            # commands have bee iterated through, add it now.
            if len(hyperedge) > 1 or not two_qubit_gate_found:
                self.add_hyperedge(hyperedge)

        # TODO: Currently we are not supporting teleportation within our
        # distributors, so we assert that all hyperedges have weight one.
        # This should be guaranteed due to `dqc_gateset` only containing
        # gates that may be implemented via EJPP packing.
        assert self.weight_one_predicate()

    def _vertex_id_predicate(self) -> bool:
        """Tests that the vertices in the hypergraph are numbered with all
        qubit vertices first and then the gate vertices in the same order
        of occurrence in the circuit. Should be guaranteed by construction.
        """
        vertices = sorted(self.vertex_list)
        qubit_vertices = sorted(self.get_qubit_vertices())

        for q in qubit_vertices:
            # If there are no vertices left or the next one is not a qubit,
            # the predicate is unsatisfied
            if not vertices or vertices.pop(0) != q:
                return False

        for cmd in self._circuit.get_commands():
            if cmd.op.type in [OpType.CU1, OpType.CZ, OpType.CX]:
                # If all vertices have been popped, unsatisfied
                if not vertices:
                    return False
                # Pop the next vertex and check commands match
                gate_vertex = vertices.pop(0)
                if self._vertex_circuit_map[gate_vertex]["command"] != cmd:
                    return False

        # There should be no more vertices left
        return not vertices

    def _get_server_to_qubit_vertex(
        self, placement: Placement
    ) -> dict[int, list[int]]:
        """Return dictionary mapping servers to a list of the qubit
        vertices which it contains.

        :param placement: Placement of hypergraph vertices onto servers.
        :type placement: Placement
        :raises Exception: Raised if the placement is not valid.
        :return: Dictionary mapping servers to a list of the qubit
        vertices which it contains.
        :rtype: dict[int, list[int]]
        """

        # Initial check that placement is valid
        if not self.is_placement(placement):
            raise Exception("This is not a valid placement for this circuit.")

        # A dictionary mapping servers to the qubit vertices it contains
        return {
            server: [
                vertex
                for vertex in self.get_qubit_vertices()
                if placement.placement[vertex] == server
            ]
            for server in set(placement.placement.values())
        }

    def to_relabeled_registers(self, placement: Placement) -> Circuit:
        """Relabel qubits to match their placement.

        :param placement: Placement of hypergraph vertices onto servers.
        :type placement: Placement
        :raises Exception: Raised if the placement is not valid.
        :return: Circuit with qubits relabeled to match servers.
        :rtype: Circuit
        """

        if not self.is_placement(placement):
            raise Exception("This is not a valid placement for this circuit.")

        server_to_vertex_dict = self._get_server_to_qubit_vertex(placement)

        circ = Circuit()
        # Map from servers to the qubit registers it contains.
        server_to_register = {}
        # Add registers to new circuit.
        for server, vertex_list in server_to_vertex_dict.items():
            server_to_register[server] = circ.add_q_register(
                f'server_{server}', len(vertex_list)
            )

        # Build map from circuit qubits to server registers
        qubit_qubit_map = {}
        for server, register in server_to_register.items():
            for i, qubit_vertex in enumerate(server_to_vertex_dict[server]):
                qubit_qubit_map[
                    self._vertex_circuit_map[qubit_vertex]["node"]
                ] = register[i]

        # Rebuild circuit using mapping from circuit qubits to server
        # registers.
        for gate in self._circuit.get_commands():
            circ.add_gate(
                gate.op,
                [qubit_qubit_map[orig_qubit] for orig_qubit in gate.args],
            )

        return circ

    def to_pytket_circuit(
        self, placement: Placement, network: NISQNetwork,
    ) -> Circuit:
        """Convert circuit to one including required distributed gates.

        :param placement: Placement of hypergraph vertices onto servers.
        :type placement: Placement
        :param network: Network on which the distributed circuit is to be run.
        :type network: NISQNetwork
        :raises Exception: Raised if the placement is not valid.
        :return: Circuit including distributed gates.
        :rtype: Circuit
        """

        # Initial check that placement is valid
        if not placement.is_valid(self, network):
            raise Exception("This is not a valid placement for this circuit.")

        server_to_qubit_vertex_list = self._get_server_to_qubit_vertex(
            placement
        )

        for server in network.get_server_list():
            if server not in server_to_qubit_vertex_list.keys():
                server_to_qubit_vertex_list[server] = []

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
        # TODO: There are several ways to tidy this portion of code:
        #   1 - Link qubit registers, rather than a separate register for
        #       each link qubit.
        #   2 - It would be cleaner to iterate through the hyperedges, rather
        #       then the servers, when adding link qubits.
        for server, qubit_vertex_list in server_to_qubit_vertex_list.items():

            # Add a register for all of the qubits assigned to this server.
            server_to_register[server] = circ.add_q_register(
                f'server_{server}', len(qubit_vertex_list)
            )

            server_to_link_register[server] = {}
            # For each hyperedge, add the necessary link qubits
            for index, hyperedge in enumerate(self.hyperedge_list):

                # List of gate vertices in this hyperedge
                gate_vertex_list = self.get_gate_vertices(hyperedge)

                # Find the one unique vertex in the hyperedge which
                # corresponds to a qubit.
                hyperedge_qubit_vertex = self.get_qubit_vertex(hyperedge)

                # Add a link qubits if the qubit of the hyperedge is not
                # placed in this server, but this server does feature in the
                # distribution tree for this hyperedge.
                if not (placement.placement[hyperedge_qubit_vertex] == server):

                    dist_tree = placement.get_distribution_tree(
                        hyperedge.vertices, hyperedge_qubit_vertex, network
                    )
                    unique_server_used = set(
                        server for edge in dist_tree for server in edge
                    )

                    if server in unique_server_used:
                        register = circ.add_q_register(
                            f'server_{server}_link_edge_{index}', 1
                        )
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
            self._vertex_circuit_map[qubit_vertex]["node"]: qubit
            for qubit_vertex, qubit in qubit_vertex_to_server_qubit.items()
        }

        # Dictionary mapping circuit qubits to hypergraph vertices
        circuit_qubit_to_vertex = {
            info["node"]: vertex
            for vertex, info in self._vertex_circuit_map.items()
            if info["type"] == "qubit"
        }

        new_command_list = self._commands.copy()
        # For each command, identify the qubits in the distributed circuit
        # which should be used. This will be the qubits which correspond
        # to that on which it originally acted if the gate and qubit are
        # in the same server. Otherwise it will be the appropriate
        # link qubit.
        for command_index, command in enumerate(new_command_list):
            # Get original circuit qubits used by command
            orig_circuit_qubit = cast(Command, command["command"]).qubits

            # Get servers to which qubits belongs
            orig_server = [
                placement.placement[circuit_qubit_to_vertex[qubit]]
                for qubit in orig_circuit_qubit
            ]

            # The server in which the gate is acted
            if command["type"] == "distributed gate":
                gate_server = placement.placement[cast(int, command["vertex"])]
            elif command["type"] == "1q local gate":
                qubit_vertex = circuit_qubit_to_vertex[orig_circuit_qubit[0]]
                gate_server = placement.placement[qubit_vertex]
            else:
                raise Exception("Command type not recognised")

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
                    if command["type"] == "distributed gate":
                        gate_vertex = command["vertex"]
                        for i, hyperedge in enumerate(self.hyperedge_list):
                            if (
                                gate_vertex in hyperedge.vertices
                                and qubit_vertex in hyperedge.vertices
                            ):
                                new_qubit.append(
                                    server_to_link_register[gate_server][i]
                                )
                    elif command["type"] == "1q local gate":
                        new_qubit.append(circuit_qubit_to_server_qubit[qubit])
                    else:
                        raise Exception("This command type is not recognised")

            assert len(orig_server) == len(new_qubit)

            # Update the dictionary from gate vertices to commands
            new_command_list[command_index]["args"] = new_qubit
            new_command_list[command_index]["op"] = cast(
                Command, command["command"]
            ).op

        # For each hyperedge add the necessary distributed operations.
        for edge_index, edge in enumerate(self.hyperedge_list):

            # List of the subset of vertices which correspond to gates.
            gate_vertex_list = self.get_gate_vertices(edge)

            # Find the qubit vertex in the hyperedge.
            qubit_vertex = self.get_qubit_vertex(edge)

            qubit_server = placement.placement[qubit_vertex]

            first_found = False
            first = 0

            # Look through the list of commands to find the first gate in
            # the hyperedge
            while not first_found:
                if new_command_list[first]["type"] == "distributed gate":

                    if new_command_list[first]["vertex"] in gate_vertex_list:
                        first_found = True
                    else:
                        first += 1
                else:
                    first += 1

            dist_tree = placement.get_distribution_tree(
                edge.vertices, qubit_vertex, network
            )

            # For every server used by hyperedge distribution tree, add a
            # starting process or teleportation before all of the gates
            # are acted. Iteration is in reverse so that the last insertion
            # ends up in first position. In this way, later commands in the
            # list are pushed after the first command.
            for distribution_edge in reversed(dist_tree):
                if qubit_server in distribution_edge:
                    args = [
                        qubit_vertex_to_server_qubit[qubit_vertex],
                        server_to_link_register[distribution_edge[1]][
                            edge_index
                        ],
                    ]
                else:
                    args = [
                        server_to_link_register[distribution_edge[0]][
                            edge_index
                        ],
                        server_to_link_register[distribution_edge[1]][
                            edge_index
                        ],
                    ]
                # TODO: I don't know if this is best practice for typing.
                # Command and int are included here almost unnecessarily.
                new_cmd: dict[str, Union[str, list[Qubit], Command, int]] = {
                    "args": args
                }
                if edge.weight == 1:
                    new_cmd["type"] = "start"
                elif edge.weight == 2:
                    new_cmd["type"] = "teleport"
                else:
                    raise Exception(
                        "The operation for this weight is not known"
                    )
                new_command_list.insert(first, new_cmd)

            last_found = False
            last = len(new_command_list) - 1

            # Look through the list of commands to find the last gate in
            # the hyperedge
            while not last_found:
                if new_command_list[last]["type"] == "distributed gate":
                    if new_command_list[last]["vertex"] in gate_vertex_list:
                        last_found = True
                    else:
                        last -= 1
                else:
                    last -= 1

            # For every server used by hyperedge distribution tree, add an
            # ending process or teleportation after all of the gates are acted.
            # Iteration is forwards through edge list in this case. In this
            # way the first starting process is undone last.
            for distribution_edge in dist_tree:
                if qubit_server in distribution_edge:
                    args = [
                        server_to_link_register[distribution_edge[1]][
                            edge_index
                        ],
                        qubit_vertex_to_server_qubit[qubit_vertex],
                    ]
                else:
                    args = [
                        server_to_link_register[distribution_edge[1]][
                            edge_index
                        ],
                        server_to_link_register[distribution_edge[0]][
                            edge_index
                        ],
                    ]
                new_cmd = {"args": args}
                if edge.weight == 1:
                    new_cmd["type"] = "end"
                elif edge.weight == 2:
                    new_cmd["type"] = "teleport"
                else:
                    raise Exception(
                        "The operation for this weight is not known"
                    )
                new_command_list.insert(last + 1, new_cmd)

        # For each command in the new command list, add it to the circuit.
        for command in new_command_list:

            if command["type"] in ["distributed gate", "1q local gate"]:
                circ.add_gate(command["op"], command["args"])
            elif command["type"] == "start":
                circ.add_custom_gate(start_proc, [], command["args"])
            elif command["type"] == "end":
                circ.add_custom_gate(end_proc, [], command["args"])
            elif command["type"] == "teleport":
                circ.add_custom_gate(telep_proc, [], command["args"])
            else:
                raise Exception("This role has not been defined")

        # Commenting this out since placement.cost is deprecated and so is
        # this to_pytket method
        # assert _cost_from_circuit(circ) == placement.cost(self, network)

        return circ


class RandomHypergraphCircuit(HypergraphCircuit):
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
            qubit_pairs = [
                [qubits[i], qubits[i + 1]] for i in range(0, n_qubits - 1, 2)
            ]

            # Act a random 2-qubit unitary between each pair.
            for pair in qubit_pairs:

                SU4 = unitary_group.rvs(4)  # random unitary in SU4
                SU4 = SU4 / (np.linalg.det(SU4) ** 0.25)

                circ.add_unitary2qbox(Unitary2qBox(SU4), *pair)

        # Rebase to a valid gate set.
        DecomposeBoxes().apply(circ)
        DQCPass().apply(circ)

        super().__init__(circ)


class CyclicHypergraphCircuit(HypergraphCircuit):
    """Particular instance of the HypergraphCircuit class, where the circuit
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
            for qubit in range(n_qubits - 1):
                circ.CZ(qubit, qubit + 1)
            circ.CZ(n_qubits - 1, 0)

        DQCPass().apply(circ)
        super().__init__(circ)


class RegularGraphHypergraphCircuit(HypergraphCircuit):
    """HypergraphCircuit constructed by acting CZ gates between qubits which
    neighbour each other in a random regular graph.
    """

    def __init__(
        self, n_qubits: int, degree: int, n_layers: int, seed: int = None,
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
                degree, n_qubits, seed=seed
            )
            for v, u in G.edges():
                circ.CZ(v, u)
                circ.Rx(random.uniform(0, 2), random.choice(list(G.nodes)))
                circ.Rz(random.uniform(0, 2), random.choice(list(G.nodes)))

        DQCPass().apply(circ)
        super().__init__(circ)
