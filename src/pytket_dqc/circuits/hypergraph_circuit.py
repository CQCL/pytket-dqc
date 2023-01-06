from __future__ import annotations

from .hypergraph import Hypergraph, Hyperedge, Vertex
from pytket import OpType, Circuit, Qubit
from pytket.circuit import Command, Op, Unitary2qBox  # type: ignore
from scipy.stats import unitary_group  # type: ignore
import numpy as np
from pytket.passes import DecomposeBoxes  # type: ignore
import networkx as nx  # type: ignore
import random
from pytket_dqc.utils import (
    dqc_gateset_predicate,
    DQCPass,
)
from pytket_dqc.utils.gateset import to_euler_with_two_hadamards

from typing import TYPE_CHECKING, Union, Optional

if TYPE_CHECKING:
    from pytket_dqc import Placement


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
        """Initialisation function

        :param circuit: Circuit to be distributed.
        :type circuit: Circuit
        """

        self.reset(circuit)

        assert self._vertex_id_predicate()
        assert self._sorted_hedges_predicate()

    def __str__(self):
        out_string = super().__str__()
        out_string += f"\nVertex Circuit Map: {self._vertex_circuit_map}"
        out_string += "\nCircuit: " + self._circuit.__str__()
        return out_string

    def to_dict(self):

        hypergraph_circuit_dict = super().to_dict()
        hypergraph_circuit_dict['circuit'] = self._circuit.to_dict()
        return hypergraph_circuit_dict

    @classmethod
    def from_dict(cls, hypergraph_circuit_dict):
        hypergraph_circuit = cls(
            Circuit.from_dict(hypergraph_circuit_dict['circuit'])
        )
        hypergraph_circuit.vertex_list = []
        hypergraph_circuit.hyperedge_list = []
        hypergraph_circuit.hyperedge_dict = dict()
        hypergraph_circuit.vertex_neighbours = dict()
        hypergraph_circuit.add_vertices(hypergraph_circuit_dict['vertex_list'])
        for hyperedge_dict in hypergraph_circuit_dict['hyperedge_list']:
            hyperedge = Hyperedge.from_dict(hyperedge_dict)
            hypergraph_circuit.add_hyperedge(
                vertices=hyperedge.vertices,
                weight=hyperedge.weight,
            )
        return hypergraph_circuit

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
        hyperedge_list_index: Optional[int] = None,
        hyperedge_dict_index: Optional[list[int]] = None,
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
                f"The first element of {vertices} "
                + "is required to be a qubit vertex."
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
            hyperedge_dict_index=hyperedge_dict_index,
        )

    def merge_hyperedge(
        self,
        to_merge_hyperedge_list: list[Hyperedge]
    ) -> Hyperedge:
        """Wrapper around `Hyperedge.merge_hyperedge` adding some checks.
        Adds no additional functionality.
        """

        new_hyperedge = super().merge_hyperedge(
            to_merge_hyperedge_list=to_merge_hyperedge_list
        )
        self._sorted_hedges_predicate()

        return new_hyperedge

    def split_hyperedge(
        self,
        old_hyperedge: Hyperedge,
        new_hyperedge_list: list[Hyperedge]
    ):
        """Wrapper around `Hyperedge.split_hyperedge` adding some checks.
        Adds no additional functionality.
        """

        super().split_hyperedge(
            old_hyperedge=old_hyperedge,
            new_hyperedge_list=new_hyperedge_list
        )

        self._sorted_hedges_predicate()

    def add_qubit_vertex(self, vertex: Vertex, qubit: Qubit):
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

    def get_qubit_vertices(self) -> list[Vertex]:
        """Return list of vertices which correspond to qubits

        :return: list of vertices which correspond to qubits
        :rtype: List[int]
        """
        return [
            vertex
            for vertex in self.vertex_list
            if self._vertex_circuit_map[vertex]["type"] == "qubit"
        ]

    def add_gate_vertex(self, vertex: Vertex, command: Command):
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

    def is_qubit_vertex(self, vertex: Vertex) -> bool:
        """Checks if the given vertex corresponds to a qubit.

        :param vertex: Vertex to be checked.
        :type vertex: int
        :return: Is it a qubit vertex.
        :rtype: bool
        """
        return self._vertex_circuit_map[vertex]["type"] == "qubit"

    def get_qubit_vertex(self, hyperedge: Hyperedge) -> Vertex:
        """Returns the qubit vertex in ``hyperedge``."""
        qubit_list = [
            vertex
            for vertex in hyperedge.vertices
            if self.is_qubit_vertex(vertex)
        ]

        assert len(qubit_list) == 1
        return qubit_list[0]

    def get_vertex_of_qubit(self, qubit: Qubit) -> Vertex:
        """Returns the vertex that corresponds to ``qubit``."""
        vertex_list = [
            vertex
            for vertex in self.vertex_list
            if self.is_qubit_vertex(vertex)
            if self._vertex_circuit_map[vertex]["node"] == qubit
        ]

        assert len(vertex_list) == 1
        return vertex_list[0]

    def get_gate_vertices(self, hyperedge: Hyperedge) -> list[Vertex]:
        """Returns the list of gate vertices in ``hyperedge``."""
        gate_vertex_list = [
            vertex
            for vertex in hyperedge.vertices
            if self._vertex_circuit_map[vertex]["type"] == "gate"
        ]

        assert len(gate_vertex_list) == len(hyperedge.vertices) - 1
        return gate_vertex_list

    def get_qubit_of_vertex(self, vertex: Vertex) -> Qubit:
        """Returns the qubit the hypergraph's vertex corresponds to.
        If the vertex is not a qubit vertex, an exception is raised.
        """
        if self._vertex_circuit_map[vertex]['type'] != 'qubit':
            raise Exception("Not a qubit vertex!")
        return self._vertex_circuit_map[vertex]['node']

    def get_gate_of_vertex(self, vertex: Vertex) -> Command:
        """Returns the gate the hypergraph's vertex corresponds to.
        If the vertex is not a gate vertex, an exception is raised.
        """
        if self._vertex_circuit_map[vertex]['type'] != 'gate':
            raise Exception("Not a gate vertex!")
        return self._vertex_circuit_map[vertex]['command']

    def get_hyperedges_containing(
        self, vertices: list[Vertex]
    ) -> list[Hyperedge]:
        """Returns the list of hyperedges that contain ``vertices`` as
        a subset of their own vertices.
        """
        return [
            hedge
            for hedge in self.hyperedge_list
            if all(v in hedge.vertices for v in vertices)
        ]

    def get_hyperedge_subcircuit(self, hyperedge: Hyperedge) -> list[Command]:
        """Returns the list of commands between the first and last gate within
        the hyperedge. Commands that don't act on the qubit vertex are omitted
        but embedded gates within the hyperedge are included.

        NOTE: The single qubit gates on the hyperedge's qubit are replaced by
        their Euler decomposition using ``to_euler_with_two_hadamards`` when
        necessary to satisfy the embedding requirements.
        NOTE: Rz gates at either side of an H-embedded CU1 gate are squashed
        together. The resulting phase must be an integer.
        """
        hyp_q_vertex = self.get_qubit_vertex(hyperedge)
        hyp_qubit = self.get_qubit_of_vertex(hyp_q_vertex)
        gate_vertices = sorted(self.get_gate_vertices(hyperedge))
        if not gate_vertices:
            return []

        def prepare_h_embedding(embed_cmds: list[Command]) -> list[Command]:
            """An auxiliary interal function to tidy up the code. Given a list
            of commands acting on ``hyp_qubit`` that start and end with a
            Hadamard, modify the 1-qubit gates accordingly so that the
            H-embedding of these commands is valid.
            In particular, we use ``to_euler_with_two_hadamards`` on each
            batch of single qubit gates and squash Rz gates at either side of
            an CU1 gate, asserting that the resulting phase must be integer.
            """
            prepared_cmds = []

            cu1_indices = [
                i
                for i, cmd in enumerate(embed_cmds)
                if cmd.op.type == OpType.CU1
            ]

            # If there are no CU1 gates, no change is required
            if not cu1_indices:
                prepared_cmds += embed_cmds
            else:
                # Include the first batch of embedded 1-qubit gates, make sure
                # that the final gate is a Hadamard
                prepared_cmds += embed_cmds[: cu1_indices[0]]
                # Remove the last Rz and remember its phase so that it may be
                # squashed with the Rz after the embedded CU1 gate
                prev_phase = 0
                if prepared_cmds[-1].op.type == OpType.Rz:
                    prev_phase = prepared_cmds[-1].op.params[0]
                    prepared_cmds.pop()  # Remove the Rz gate
                assert prepared_cmds[-1].op.type == OpType.H
                # Append the first embedded CU1 gate
                prepared_cmds.append(embed_cmds[cu1_indices[0]])

                # Append all gates from here until the last embedded CU1 gate.
                # Any batch of 1-qubit gates between CU1 gates needs to be
                # converted to an explicit Euler form [Rz,H,Rz,H] where the
                # missing Rz gate at the end has been squashed with that of
                # the next batch.
                prev_cu1_idx = cu1_indices[0]
                for next_cu1_idx in cu1_indices[1:]:
                    # Get the current batch of embedded 1-qubit gates, make
                    # sure they are of the form [Rz,H,Rz,H,Rz] and squash
                    # ``prev_phase`` into the first Rz gate.
                    current_1q_ops = [
                        cmd.op
                        for cmd in embed_cmds[
                            prev_cu1_idx + 1 : next_cu1_idx  # noqa: E203
                        ]
                    ]
                    new_ops = to_euler_with_two_hadamards(current_1q_ops)
                    assert len(new_ops) == 5  # [Rz,H,Rz,H,Rz]

                    first_rz = new_ops[0]
                    assert first_rz.type == OpType.Rz
                    squashed_phase = prev_phase + first_rz.params[0]

                    middle_rz = new_ops[2]
                    assert middle_rz.type == OpType.Rz
                    mid_phase = middle_rz.params[0]

                    last_rz = new_ops[4]
                    assert last_rz.type == OpType.Rz
                    last_phase = last_rz.params[0]

                    # If the middle Rz gate has phase k2pi, we can push the
                    # squashed phase to the end, merge it with the last
                    # and leave this as the ``prev_phase`` for next iteration
                    if np.isclose(mid_phase % 2, 0) or np.isclose(
                        mid_phase % 2, 2
                    ):
                        prev_phase = squashed_phase + mid_phase + last_phase
                        squashed_phase = 0
                    # Otherwise, ``squashed_phase`` must be an integer and
                    # we store the phase of the last CZ for the next iteration
                    else:
                        assert np.isclose(squashed_phase % 1, 0) or np.isclose(
                            squashed_phase % 1, 1
                        )
                        prev_phase = last_phase

                    # Replace the phase of the first Rz with ``squashed_phase``
                    new_ops[0] = Op.create(OpType.Rz, squashed_phase)
                    # Create the command list
                    current_1q_cmds = [
                        Command(op, [hyp_qubit]) for op in new_ops
                    ]
                    # Remove last Rz; its phase is stored in ``prev_phase``
                    current_1q_cmds.pop()
                    # Append the batch of embedded 1-qubit gates [Rz,H,Rz,H]
                    prepared_cmds += current_1q_cmds
                    # Append the next embedded CU1 gate
                    prepared_cmds.append(embed_cmds[next_cu1_idx])
                    prev_cu1_idx = next_cu1_idx

                # Include the last batch of embedded 1-qubit gates, make sure
                # that the first gate is an Rz and that ``prev_phase`` is
                # squashed into it
                last_1q_cmds = embed_cmds[prev_cu1_idx + 1 :]  # noqa: E203
                if last_1q_cmds[0].op.type == OpType.Rz:
                    rz = last_1q_cmds.pop(0)  # Remove it
                    prev_phase += rz.op.params[0]  # Squash phases together
                # Sanity check: the phase is an integer
                assert np.isclose(prev_phase % 1, 0) or np.isclose(
                    prev_phase % 1, 1
                )
                # Append the Rz gate with the squashed phase
                rz = Command(Op.create(OpType.Rz, prev_phase), [hyp_qubit])
                prepared_cmds.append(rz)
                # Append the rest of the embedded commands
                prepared_cmds += last_1q_cmds

            return prepared_cmds

        subcirc_commands = []
        subcirc_commands.append(self.get_gate_of_vertex(gate_vertices[0]))
        prev_hyp_gate_vertex = gate_vertices[0]

        for next_hyp_gate_vertex in gate_vertices[1:]:
            # Get all embedded gates between the previous and next distributed
            # gates. Omit any commands that do not act on ``hyp_qubit``.
            embedded_commands = self.get_intermediate_commands(
                prev_hyp_gate_vertex, next_hyp_gate_vertex, hyp_q_vertex
            )

            h_indices = [
                i
                for i, cmd in enumerate(embedded_commands)
                if cmd.op.type == OpType.H
            ]
            # If there are no Hadamards, then it's all a D-embedding
            if not h_indices:
                # We don't need to change anything
                subcirc_commands += embedded_commands
            else:
                # There must be at least two Hadamards, one starting
                # the H-embedding and the other ending it
                assert len(h_indices) >= 2
                # All embedded commands before the first Hadamard
                # are D-embedded
                subcirc_commands += embedded_commands[: h_indices[0]]
                # All embedded commands between the two Hadamard gates
                # are H-embedded. These may need their 1-qubit gates to
                # be changed, decomposing Hadamards and squashing phases
                subcirc_commands += prepare_h_embedding(
                    embedded_commands[
                        h_indices[0] : h_indices[-1] + 1  # noqa: E203
                    ]
                )
                # All embedded commands after the last Hadamard
                # are D-embedded
                subcirc_commands += embedded_commands[
                    h_indices[-1] + 1 :  # noqa: E203
                ]

            # Now that all embedded gates have been added, append the next
            # distributed gate and continue with the loop
            subcirc_commands.append(
                self.get_gate_of_vertex(next_hyp_gate_vertex)
            )
            prev_hyp_gate_vertex = next_hyp_gate_vertex

        return subcirc_commands

    def requires_h_embedded_cu1(self, hyperedge: Hyperedge) -> bool:
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
            raise Exception(
                "The inputted circuit is not in a valid gateset. "
                + "You can apply ``DQCPass`` from pytket_dqc.utils "
                + "on the circuit to rebase it to a valid gateset."
            )

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

    def _sorted_hedges_predicate(self) -> bool:
        """Tests that the hyperedges are in circuit sequential order
        in `self.hyperedge_list`.
        """

        for qubit_vertex in self.get_qubit_vertices():
            hedge_list = [
                hedge
                for hedge in self.hyperedge_list
                if self.get_qubit_vertex(hedge) == qubit_vertex
            ]
            if len(hedge_list) <= 1:
                continue
            if hedge_list != sorted(
                hedge_list,
                key=lambda hedge: min(
                    [v for v in hedge.vertices if v != qubit_vertex]
                ),
            ):
                return False

        return True

    def _get_server_to_qubit_vertex(
        self, placement: Placement
    ) -> dict[int, list[Vertex]]:
        """Return dictionary mapping servers to a list of the qubit
        vertices which it contains.

        :param placement: Placement of hypergraph vertices onto servers.
        :type placement: Placement
        :raises Exception: Raised if the placement is not valid.
        :return: Dictionary mapping servers to a list of the qubit
        vertices which it contains.
        :rtype: dict[int, list[Vertex]]
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

    def get_vertex_to_command_index_map(self) -> dict[Vertex, int]:
        """Get a mapping from each gate `Vertex` in the `Hypergraph`, to its
        corresponding index in the list returned by `Circuit.get_commands()`.
        """

        vertex_to_command_index_map: dict[Vertex, int] = dict()
        for command_index, command_dict in enumerate(self._commands):
            if command_dict["type"] == "distributed gate":
                vertex = command_dict["vertex"]
                assert type(vertex) == Vertex
                vertex_to_command_index_map[vertex] = command_index
        return vertex_to_command_index_map

    def get_last_gate_vertex(self, gate_vertex_list: list[Vertex]) -> Vertex:
        """Given a list of gate vertices,
        return the vertex in `Circuit.get_commands()`
        that corresponds to the last gate in the circuit.
        """

        return max(gate_vertex_list)

    def get_first_gate_vertex(self, gate_vertex_list: list[Vertex]) -> Vertex:
        """Given a list of gate vertices,
        return the vertex in `Circuit.get_commands()`
        that corresponds to the first gate in the circuit.
        """

        assert all(
            [v >= len(self.get_qubit_vertices()) for v in gate_vertex_list]
        )

        return min(gate_vertex_list)

    def get_intermediate_commands(
        self, first_vertex: Vertex, second_vertex: Vertex, qubit_vertex: Vertex
    ) -> list[Command]:
        """Given two gate vertices and a qubit vertex, return all commands
        in the circuit after the gate corresponding to ``first_vertex`` and up
        until the gate corresponding to ``second_vertex``.

        NOTE: the ``first_vertex`` and ``second_vertex`` gates aren't included
        NOTE: only the commands acting on ``qubit_vertex`` are included.
        """

        assert self.is_qubit_vertex(qubit_vertex)
        assert first_vertex in self.vertex_list and not self.is_qubit_vertex(
            first_vertex
        )
        assert second_vertex in self.vertex_list and not self.is_qubit_vertex(
            second_vertex
        )

        qubit = self.get_qubit_of_vertex(qubit_vertex)

        vertex_to_command_index_map = self.get_vertex_to_command_index_map()
        first_command_index = vertex_to_command_index_map[first_vertex]
        second_command_index = vertex_to_command_index_map[second_vertex]

        intermediate_commands = []

        for command_dict in self._commands[
            first_command_index + 1 : second_command_index  # noqa: E203
        ]:
            command = command_dict["command"]
            assert type(command) == Command
            if qubit in command.qubits:
                intermediate_commands.append(command_dict["command"])

        return intermediate_commands

    def is_h_embeddable_CU1(
        self, command: Command, servers: set[int], placement: Placement
    ) -> bool:
        """Check that a CU1 could be embeddable in a H embedding unit.

        A return value of ``False`` only proves it cannot be embedded.
        Since we do not check the 1q gates surrounding it,
        a return value of ``True`` does not guarantee it is embeddable in
        a H embedding unit.
        """
        assert command.op.type == OpType.CU1

        this_command_qubit_vertices = [
            self.get_vertex_of_qubit(qubit)
            for qubit in command.qubits
        ]

        this_command_servers = {
            placement.placement[qubit_vertex]
            for qubit_vertex in this_command_qubit_vertices
        }

        if servers != this_command_servers:
            return False

        return (
            np.isclose(command.op.params[0] % 1, 0)
            or np.isclose(command.op.params[0] % 1, 1)
        )


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
        self, n_qubits: int, degree: int, n_layers: int, **kwargs,
    ):
        """Initialisation function

        :param n_qubits: The number of qubits on which the circuit acts.
        :type n_qubits: int
        :param degree: The degree of the random regular graph.
        :type degree: int
        :param n_layers: The number of random regular graphs to generate.
        :type n_layers: int

        :key seed: Seed for the random generation of regular graphs,
            defaults to None
        :type seed: int, optional
        """

        seed = kwargs.get("seed", None)

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
