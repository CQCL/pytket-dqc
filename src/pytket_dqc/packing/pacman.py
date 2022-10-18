from __future__ import annotations  # May be redundant in future (PEP 649)
import logging
from numpy import isclose, bool_
import networkx as nx  # type: ignore
from networkx.algorithms import bipartite  # type: ignore
from typing import List, Dict, Tuple, Optional

from pytket_dqc.circuits import HypergraphCircuit, Hyperedge, Vertex
from pytket_dqc.placement import Placement
from pytket.circuit import Command, OpType, Op, Qubit  # type: ignore
from pytket_dqc.utils import (
    is_distributable,
    distributable_1q_op_types,
    distributable_op_types,
)

logging.basicConfig(level=logging.INFO)


class Packet:
    """The basic structure of a collection of
    distributable gates.
    """

    def __init__(
        self,
        packet_index: int,
        qubit_vertex: Vertex,
        connected_server_index: int,
        gate_vertices: List[Vertex],
        contained_embeddings: List[Tuple[Packet]] = list(),
    ):
        self.packet_index: int = packet_index
        self.qubit_vertex: Vertex = qubit_vertex
        self.connected_server_index: int = connected_server_index
        self.gate_vertices: List[Vertex] = gate_vertices

    def __str__(self):
        return f"P{self.packet_index}"

    def __repr__(self):
        return self.__str__()

    def __eq__(self, o):
        if isinstance(o, Packet):
            return (
                o.packet_index == self.packet_index
                and o.qubit_vertex == self.qubit_vertex
                and o.gate_vertices == self.gate_vertices
                and o.connected_server_index == self.connected_server_index
            )
        return False

    def __hash__(self):
        return hash(repr(self))


class PacMan:
    """Builds and manages packets using a HypergraphCircuit and a Placement"""

    def __init__(
        self, hypergraph_circuit: HypergraphCircuit, placement: Placement
    ):
        self.hypergraph_circuit = hypergraph_circuit
        self.placement = placement
        self.vertex_to_command_index: Dict[Vertex, int] = dict()
        self.packets_by_qubit: Dict[Vertex, List[Packet]] = dict()
        self.hopping_packets: Dict[Vertex, List[Tuple[Packet, ...]]] = dict()
        self.neighbouring_packets: Dict[
            Vertex, List[Tuple[Packet, ...]]
        ] = dict()
        self.merged_packets: Dict[Vertex, List[Tuple[Packet, ...]]] = dict()

        self.build_vertex_to_command_index()
        self.build_packets()
        self.identify_neighbouring_packets()
        self.identify_hopping_packets()
        self.merge_all_packets()

    def build_vertex_to_command_index(self):
        """For each vertex in the hypergraph,
        find its corresponding index in the list
        returned by Circuit.get_commands()

        QUESTION: Can/should I make this a `HypergraphCircuit` method instead?
        """
        # TODO: Rewrite this the way Pablo intended
        for command_index, command_dict in enumerate(
            self.hypergraph_circuit._commands
        ):
            if command_dict["type"] == "distributed gate":
                vertex = command_dict["vertex"]
                self.vertex_to_command_index[vertex] = command_index

    def build_packets(self):
        """Build packets from `Hyperedges`
        Essentially split them up if they go to different servers

        QUESTION: Would it be better to have each CU1 actually be its
        own packet (I think this is what Junyi explicitly asks for)?
        """
        hyperedges_ordered = self.get_hyperedges_ordered()
        starting_index = 0
        for qubit_vertex in self.hypergraph_circuit.get_qubit_vertices():
            self.packets_by_qubit[qubit_vertex] = []
            for hyperedge in hyperedges_ordered[qubit_vertex]:
                starting_index, packets = self.hyperedge_to_packets(
                    hyperedge, starting_index
                )
                for packet in packets:
                    self.packets_by_qubit[qubit_vertex].append(packet)

    def identify_neighbouring_packets(self):
        for qubit_vertex in self.hypergraph_circuit.get_qubit_vertices():
            logging.debug(
                f"Checking qubit vertex {qubit_vertex} \
                    for neighbouring packets."
            )
            neighbouring_packets: List[Tuple[Packet]] = list()
            considered_packet_list = []
            for packet in self.packets_by_qubit[qubit_vertex][:-1]:
                if packet in considered_packet_list:
                    continue
                neighbouring_packet = [packet]
                next_packet = self.get_next_packet(packet)
                while (
                    next_packet is not None
                    and self.are_neighbouring_packets(packet, next_packet)
                ):
                    neighbouring_packet.append(next_packet)
                    considered_packet_list.append(next_packet)
                    next_packet = self.get_next_packet(next_packet)
                    neighbouring_packets.append(tuple(neighbouring_packet))
            self.neighbouring_packets[qubit_vertex] = neighbouring_packets

    def identify_hopping_packets(self):
        for qubit_vertex in self.hypergraph_circuit.get_qubit_vertices():
            logging.debug(
                f"Checking qubit vertex {qubit_vertex} for hopping packets."
            )
            self.hopping_packets[qubit_vertex] = []
            for packet in self.packets_by_qubit[qubit_vertex][:-1]:
                next_packet = self.get_next_packet(packet)
                while (
                    next_packet is not None
                    and self.get_next_packet(next_packet) is not None
                ):
                    next_packet = self.get_next_packet(next_packet)
                    logging.debug(f"Comparing {packet} and {next_packet}")
                    if self.are_hoppable_packets(packet, next_packet):
                        self.hopping_packets[qubit_vertex].append(
                            (packet, next_packet)
                        )
                        break

    def merge_all_packets(self):
        """Using the identified neighbouring and hopping
        packets, create tuples merging them together
        """
        logging.debug("Merging packets")
        for qubit_vertex in self.hypergraph_circuit.get_qubit_vertices():
            self.merged_packets[qubit_vertex] = []
            considered_packets = []
            for packet in self.packets_by_qubit[qubit_vertex]:
                logging.debug(f"Checking packet {packet}")
                if packet in considered_packets:
                    logging.debug("Already checked")
                    continue
                mergeable_packets = [packet]
                continue_merging = True
                while continue_merging:
                    logging.debug(f"Start of checks {mergeable_packets}")
                    logging.debug(f"Packet of interest is {packet}")
                    if (
                        self.get_subsequent_neighbouring_packet(packet)
                        is not None
                    ):
                        packet = self.get_subsequent_neighbouring_packet(
                            packet
                        )
                        logging.debug(
                            f"Adding {packet} to mergeable \
                                packets (neighbouring)"
                        )
                        mergeable_packets.append(packet)
                        considered_packets.append(packet)
                    elif (
                        self.get_subsequent_hopping_packet(packet) is not None
                    ):
                        packet = self.get_subsequent_hopping_packet(packet)
                        logging.debug(
                            f"Adding {packet} to mergeable packets (hopping) \
                                {mergeable_packets}"
                        )
                        mergeable_packets.append(packet)
                        considered_packets.append(packet)
                    else:
                        continue_merging = False
                    logging.debug(f"End of checks {mergeable_packets}")

                self.merged_packets[qubit_vertex].append(
                    tuple(mergeable_packets)
                )

    def get_subsequent_neighbouring_packet(
        self, packet: Packet
    ) -> Optional[Packet]:
        potential_neighbouring_packets = [
            neighbouring_packets
            for neighbouring_packets in self.neighbouring_packets[
                packet.qubit_vertex
            ]
            if neighbouring_packets[0] == packet
        ]
        assert (
            len(potential_neighbouring_packets) <= 1
        ), "There should only be up to 1 neighbouring packet \
            containing this packet."

        if len(potential_neighbouring_packets) == 0:
            return None
        else:
            return potential_neighbouring_packets[0][1]

    def get_subsequent_hopping_packet(
        self, packet: Packet
    ) -> Optional[Packet]:
        potential_hopping_packets = [
            hopping_packet
            for hopping_packet in self.hopping_packets[packet.qubit_vertex]
            if hopping_packet[0] == packet
        ]
        assert (
            len(potential_hopping_packets) <= 1
        ), "There should only be up to 1 hopping packet \
            containing this packet."

        if len(potential_hopping_packets) == 0:
            return None
        else:
            return potential_hopping_packets[0][1]

    def get_next_packet(self, packet: Packet) -> Optional[Packet]:
        """Find the next packet on the qubit of the given packet that
        is connected to the same server as the given packet.
        Returns None if no such packet exists
        """
        for potential_packet in self.packets_by_qubit[packet.qubit_vertex]:
            if potential_packet.packet_index <= packet.packet_index:
                continue

            if (
                potential_packet.connected_server_index
                == packet.connected_server_index
            ):
                return potential_packet

        return None

    def get_all_packets(self) -> List[Packet]:
        """Return a list of all packets on the entire circuit"""
        all_packets: List[Packet] = []
        for packets in self.packets_by_qubit.values():
            all_packets += packets
        all_packets.sort(key=lambda x: x.packet_index)
        return all_packets

    def are_neighbouring_packets(
        self, first_packet: Packet, second_packet: Packet
    ):
        """Check if two given packets are neighbouring packets"""

        if (
            first_packet.qubit_vertex != second_packet.qubit_vertex
            or first_packet.connected_server_index
            != second_packet.connected_server_index
        ):
            return False

        intermediate_commands = self.get_intermediate_commands(
            first_packet, second_packet
        )
        for command in intermediate_commands:
            if not is_distributable(command.op):
                return False

        return True

    def are_hoppable_packets(
        self, first_packet: Packet, second_packet: Packet
    ) -> bool:
        """Check if two given packets are hopping packets"""

        logging.debug(
            f"Are packets {first_packet} and "
            + f"{second_packet} packable via embedding?"
        )

        allowed_op_types = distributable_op_types + [OpType.H]

        intermediate_commands = self.get_intermediate_commands(
            first_packet, second_packet
        )
        logging.debug(
            f"Intermediate ops \
                {[command.op for command in intermediate_commands]}"
        )
        cu1_indices = []
        for i, command in enumerate(intermediate_commands):
            if command.op.type not in allowed_op_types:
                logging.debug(f"No, {command} is not embeddable.")
                return False
            if command.op.type == OpType.CU1:
                if not self.is_embeddable_CU1(command, first_packet):
                    logging.debug(
                        f"No, CU1 command {command} is not embeddable."
                    )
                    return False
                cu1_indices.append(i)

        ops_1q_list: List[List[Op]] = []

        # Convert the intermediate commands between CU1s as necessary
        first_commands = intermediate_commands[: cu1_indices[0]]
        first_ops = [command.op for command in first_commands]
        ops_1q_list.append(first_ops)
        for i, cu1_index in enumerate(cu1_indices):
            if cu1_index == cu1_indices[0]:
                continue
            commands = intermediate_commands[
                cu1_indices[i - 1] + 1: cu1_index
            ]
            ops = [command.op for command in commands]
            if OpType.H in [op.type for op in ops]:
                ops_1q_list.append(self.convert_1q_ops(ops))
            else:
                ops_1q_list.append(ops)

        last_commands = intermediate_commands[cu1_indices[-1] + 1:]
        last_ops = [command.op for command in last_commands]
        ops_1q_list.append(last_ops)
        logging.debug(f"ops_1q_list {ops_1q_list}")

        for i, ops_1q in enumerate(ops_1q_list[:-1]):
            if i == 0:
                n_hadamards = len([op for op in ops_1q if op.type == OpType.H])
                if n_hadamards != 1:
                    logging.debug(
                        "No, there can only be 1 Hadamard "
                        + "the start of the Hadamard sandwich."
                    )
                    return False

            if i == len(ops_1q_list) - 2:
                n_hadamards = len(
                    [op for op in ops_1q_list[-1] if op.type == OpType.H]
                )
                if n_hadamards != 1:
                    logging.debug(
                        "No, there can only be 1 Hadamard "
                        + "at the end of the Hadamard sandwich."
                    )
                    return False

            if not self.are_1q_op_phases_npi(ops_1q, ops_1q_list[i + 1]):
                logging.debug(
                    f"No, the phases of {ops_1q} and "
                    + f"{ops_1q_list[i+1]} prevent embedding."
                )
                return False

        return True

    def hyperedge_to_packets(
        self, hyperedge: Hyperedge, starting_index: int
    ) -> Tuple[int, List[Packet]]:
        """Convert a hyperedge into a packet(s)
        Multiple needed if the hyperedge ends up having gates
        distributed to multiple servers
        """
        hyperedge_qubit_vertex = self.hypergraph_circuit.get_qubit_vertex(
            hyperedge
        )
        connected_server_to_dist_gates: Dict[
            int, List[Vertex]
        ] = {}  # Server number to list of distributed gates on that server.
        packets: List[Packet] = []
        for gate_vertex in self.hypergraph_circuit.get_gate_vertices(
            hyperedge
        ):
            connected_server = self.get_connected_server(
                hyperedge_qubit_vertex, gate_vertex
            )
            if connected_server in connected_server_to_dist_gates.keys():
                connected_server_to_dist_gates[connected_server].append(
                    gate_vertex
                )
            else:
                connected_server_to_dist_gates[connected_server] = [
                    gate_vertex
                ]

        for (
            connected_server,
            dist_gates,
        ) in connected_server_to_dist_gates.items():
            packets.append(
                Packet(
                    starting_index,
                    hyperedge_qubit_vertex,
                    connected_server,
                    dist_gates,
                )
            )
            starting_index += 1

        return starting_index, packets

    def get_hyperedges_ordered(self) -> Dict[Vertex, List[Hyperedge]]:
        """Orders hyperedges into their circuit sequential order

        QUESTION: Move to `HypergraphCircuit`?
        """
        hyperedges: Dict[Vertex, List[Hyperedge]] = {}

        # Populate the qubit_vertex keys
        for qubit_vertex in self.hypergraph_circuit.get_qubit_vertices():
            hyperedges[qubit_vertex] = []

        # Populate the lists with unsorted hyperedges
        for hyperedge in self.hypergraph_circuit.hyperedge_list:
            qubit_vertex = self.hypergraph_circuit.get_qubit_vertex(hyperedge)
            hyperedges[qubit_vertex].append(hyperedge)

        # Sort the hyperedges using bubble sort
        # TODO: Do it more efficiently
        for qubit_vertex in self.hypergraph_circuit.get_qubit_vertices():
            hyperedge_list: List[Hyperedge] = hyperedges[qubit_vertex]
            changes = -1  # Just a temporary storage value
            while changes != 0:
                changes = 0
                for i in range(len(hyperedge_list) - 1):
                    if self.get_last_command_index(
                        self.hypergraph_circuit.get_gate_vertices(
                            hyperedge_list[i]
                        )
                    ) > self.get_first_command_index(
                        self.hypergraph_circuit.get_gate_vertices(
                            hyperedge_list[i + 1]
                        )
                    ):
                        hyperedge_list[i], hyperedge_list[i + 1] = (
                            hyperedge_list[i + 1],
                            hyperedge_list[i],
                        )
                        changes += 1

            # Might be redundant line of code here
            hyperedges[qubit_vertex] = hyperedge_list

        return hyperedges

    def get_connected_server(self, qubit_vertex: Vertex, gate_vertex: Vertex):
        command_index = self.vertex_to_command_index[gate_vertex]
        command_dict = self.hypergraph_circuit._commands[
            command_index
        ]  # TODO: Rewrite this as Pablo intended
        command = command_dict["command"]
        assert (
            type(command) == Command
        )  # This is for mypy check - is there a better way?
        qubits = command.qubits
        qubit_vertex_candidates = [
            self.qubit_vertex_from_qubit(qubit)
            for qubit in qubits
            if self.qubit_vertex_from_qubit(qubit) is not qubit_vertex
        ]

        assert (
            len(qubit_vertex_candidates) == 1
        ), "There should be 1 and only 1 other qubit vertex."
        other_qubit_vertex = qubit_vertex_candidates[0]
        return self.placement.placement[other_qubit_vertex]

    def qubit_vertex_from_qubit(self, qubit: Qubit):
        """Given Qubit, find it's vertex on the hypergraph
        Question: Should be `HypergraphCircuit` method?
        """
        # TODO: Maybe not needed anymore
        for (
            vertex,
            circuit_element,
        ) in (
            self.hypergraph_circuit._vertex_circuit_map.items()
        ):  # TODO: Don't access hidden stuff
            if (
                circuit_element["type"] == "qubit"
                and circuit_element["node"] == qubit
            ):
                return vertex
        raise Exception("Could not find a vertex corresponding to this qubit.")

    def circuit_element_from_vertex(self, vertex: Vertex):
        """Given a vertex, return the circuit element (Qubit or Command)
        Question: Should be `HypergraphCircuit` method?
        """
        # TODO: Don't access hidden stuff
        if (
            self.hypergraph_circuit._vertex_circuit_map[vertex]["type"]
            == "qubit"
        ):
            return self.hypergraph_circuit._vertex_circuit_map[vertex]["node"]
        elif (
            self.hypergraph_circuit._vertex_circuit_map[vertex]["type"]
            == "gate"
        ):
            return self.hypergraph_circuit._vertex_circuit_map[vertex][
                "command"
            ]

    def get_last_command_index(self, gate_vertex_list: List[Vertex]) -> int:
        """Given a list of gate vertices, return the vertex that
        corresponds to the last gate in the circuit.
        Question: Should be `HypergraphCircuit` method?
        """
        last_command_index: int = -1  # Placeholder value
        for vertex in gate_vertex_list:
            if (
                last_command_index == -1
                or self.vertex_to_command_index[vertex] > last_command_index
            ):
                last_command_index = self.vertex_to_command_index[vertex]

        return last_command_index

    def get_first_command_index(self, gate_vertex_list: List[Vertex]) -> int:
        """Given a list of gate vertices, return the vertex that
        corresponds to the last gate in the circuit.
        Question: Should be `HypergraphCircuit` method?
        """
        first_command_index: int = -1  # Placeholder value
        for vertex in gate_vertex_list:
            if (
                first_command_index == -1
                or self.vertex_to_command_index[vertex] < first_command_index
            ):
                first_command_index = self.vertex_to_command_index[vertex]

        return first_command_index

    def get_intermediate_commands(
        self, first_packet: Packet, second_packet: Packet
    ) -> List[Command]:
        assert (
            first_packet.qubit_vertex == second_packet.qubit_vertex
        ), "Qubit vertices do not match."

        qubit_vertex = first_packet.qubit_vertex
        qubit = self.circuit_element_from_vertex(qubit_vertex)

        last_command_index = self.get_last_command_index(
            first_packet.gate_vertices
        )
        first_command_index = self.get_first_command_index(
            second_packet.gate_vertices
        )

        intermediate_commands = []

        for (
            command_dict
        ) in self.hypergraph_circuit._commands[  # TODO: Hidden stuff
            last_command_index + 1: first_command_index
        ]:
            command = command_dict["command"]
            assert type(command) == Command
            if qubit in command.qubits:
                intermediate_commands.append(command_dict["command"])

        return intermediate_commands

    def is_embeddable_CU1(self, command: Command, packet: Packet) -> bool:
        """Bad function name but this checks that a CU1 itself
        COULD be embeddable
        i.e. only prove that we cannot embed but does not check the 1q
        gates surrounding it, hence doesn't prove it IS embeddable
        """
        packet_servers = {
            self.placement.placement[packet.qubit_vertex],
            packet.connected_server_index,
        }

        this_command_qubit_vertices = [
            self.qubit_vertex_from_qubit(qubit) for qubit in command.qubits
        ]
        this_command_servers = {
            self.placement.placement[qubit_vertex]
            for qubit_vertex in this_command_qubit_vertices
        }
        if packet_servers != this_command_servers:
            return False
        elif not isclose(command.op.params[0], 1):
            return False
        return True

    def convert_1q_ops(self, ops: List[Op]) -> List[Op]:
        """Converts a set of 1q ops
        in the gateset of Rz, Z, X, H
        with up to 1 Hadamard
        so that it is in the same gateset
        but has 2 Hadamards in the list.
        """

        hadamard_indices = [
            i for i, op in enumerate(ops) if op.type == OpType.H
        ]
        hadamard = Op.create(OpType.H)
        hadamard_count = len(hadamard_indices)

        assert (
            hadamard_count <= 2
        ), f"There should not be more than 2 Hadamards. {ops}"

        if hadamard_count == 2:
            logging.debug(f"Returning {ops} unchanged.")
            return ops

        elif hadamard_count == 0:
            logging.debug(f"Appending 2 Hadamards to {ops}")
            ops.append(hadamard)
            ops.append(hadamard)
            logging.debug(f"Appending 2 Hadamards to {ops}")
            return ops

        else:
            assert (
                len(ops) <= 3
            ), "There can only be up to 3 ops in this decomposition."
            new_ops: List[Op] = []
            s_op = Op.create(OpType.U1, [0.5])

            if len(ops) == 1:
                new_ops += [s_op, hadamard, s_op, hadamard, s_op]

            elif len(ops) == 2:
                phase_op_index = int(
                    not hadamard_indices[0]
                )  # only takes value of 1 or 0
                phase_op = ops[phase_op_index]
                phase = phase_op.params[0]  # phase in turns of pi
                new_phase_op = Op.create(
                    OpType.U1, [phase + 1 / 2]
                )  # need to add another half phase
                new_ops += [new_phase_op, hadamard, s_op, hadamard, s_op]
                if phase_op_index:
                    new_ops.reverse()

            else:
                first_phase = ops[0].params[0]
                second_phase = ops[2].params[0]
                first_new_phase_op = Op.create(OpType.U1, [first_phase + 0.5])
                second_new_phase_op = Op.create(
                    OpType.U1, [second_phase + 0.5]
                )
                new_ops += [
                    first_new_phase_op,
                    hadamard,
                    s_op,
                    hadamard,
                    second_new_phase_op,
                ]
            logging.debug(f"Converted {ops} for {new_ops}")
            return new_ops

    def are_1q_op_phases_npi(
        self, prior_1q_ops: List[Op], post_1q_ops: List[Op]
    ) -> bool_:
        """Check if the sum of the params of the
        U1 gates that sandwich a CU1 are
        equal to n
        (i.e. the actual phases are equal to n * pi)

        QUESTION: Move to utils? It's a bit specific I suppose though
        """

        if (
            len(prior_1q_ops) == 0
            or prior_1q_ops[-1].type == OpType.H
            or all(
                [op.type in distributable_1q_op_types for op in prior_1q_ops]
            )
            and prior_1q_ops[-1].type != OpType.Rz
        ):
            prior_phase = 0

        else:
            prior_op = prior_1q_ops[-1]
            prior_phase = prior_op.params[0]

        if (
            len(post_1q_ops) == 0
            or post_1q_ops[0].type == OpType.H
            or all(
                [op.type in distributable_1q_op_types for op in post_1q_ops]
            )
            and post_1q_ops[0].type != OpType.Rz
        ):
            post_phase = 0

        else:
            post_op = post_1q_ops[0]
            post_phase = post_op.params[0]

        phase_sum = prior_phase + post_phase

        return isclose(phase_sum % 1, 0)

    def get_connected_packets(self, packet: Packet):
        connected_packets = []
        for gate_vertex in packet.gate_vertices:
            gate_qubits = self.circuit_element_from_vertex(gate_vertex).qubits
            other_qubit_candidates = [
                qubit
                for qubit in gate_qubits
                if self.qubit_vertex_from_qubit(qubit) != packet.qubit_vertex
            ]
            assert (
                len(other_qubit_candidates) == 1
            ), "There should only be one other qubit candidate \
                for this gate vertex."
            other_qubit_vertex = self.qubit_vertex_from_qubit(
                other_qubit_candidates[0]
            )
            for potential_packet in self.packets_by_qubit[other_qubit_vertex]:
                if gate_vertex in potential_packet.gate_vertices:
                    if potential_packet not in connected_packets:
                        connected_packets.append(potential_packet)
                    break
        return connected_packets

    def get_nx_graph_merged(self):
        """Get the NetworkX graph representing the circuit.
        Nodes are merged packets.
        """
        graph = nx.Graph()
        top_packets = set()
        bottom_packets = set()
        edges = set()
        for qubit_vertex in self.hypergraph_circuit.get_qubit_vertices():
            for merged_packet in self.merged_packets[qubit_vertex]:
                for (
                    connected_merged_packet
                ) in self.get_connected_merged_packets(merged_packet):
                    if (
                        tuple([merged_packet, connected_merged_packet])
                        not in edges
                    ):
                        if connected_merged_packet in top_packets:
                            assert merged_packet not in top_packets
                            bottom_packets.add(merged_packet)
                        elif connected_merged_packet in bottom_packets:
                            assert merged_packet not in bottom_packets
                            top_packets.add(merged_packet)
                        else:
                            top_packets.add(merged_packet)
                            bottom_packets.add(connected_merged_packet)
                        edges.add(
                            tuple([merged_packet, connected_merged_packet])
                        )

        graph.add_nodes_from(top_packets, bipartite=0)
        graph.add_nodes_from(bottom_packets, bipartite=1)
        graph.add_edges_from(edges)
        assert nx.is_bipartite(graph), "The graph must be bipartite."
        return graph, top_packets

    def get_containing_merged_packet(self, packet: Packet):
        for merged_packet in self.merged_packets[packet.qubit_vertex]:
            if packet in merged_packet:
                return merged_packet

    def get_connected_merged_packets(
        self, merged_packet: Tuple[Packet]
    ) -> List[Tuple[Packet]]:
        connected_merged_packets: List[Tuple[Packet]] = list()
        for packet in merged_packet:
            connected_packets = self.get_connected_packets(packet)
            for connected_packet in connected_packets:
                connected_merged_packets.append(
                    self.get_containing_merged_packet(connected_packet)
                )

        return connected_merged_packets

    def get_mvc_merged_graph(self):
        g, topnodes = self.get_nx_graph_merged()
        matching = bipartite.maximum_matching(g, top_nodes=topnodes)
        return bipartite.to_vertex_cover(g, matching, top_nodes=topnodes)

    def get_true_conflict_edges(self) -> List[Tuple[Packet, ...]]:
        cg, topnodes = self.get_nx_graph_conflict()
        mvc = self.get_mvc_merged_graph()
        true_conflicts = list()
        for u, v in cg.edges():
            if u in mvc and v in mvc:
                true_conflicts.append(tuple([u, v]))

        return true_conflicts

    def get_embedded_packets(
        self, hopping_packet: Tuple[Packet, ...]
    ) -> List[Packet]:
        initial_index = hopping_packet[0].packet_index
        final_index = hopping_packet[1].packet_index
        embedded_packets: List[Packet] = []
        for packet in self.packets_by_qubit[hopping_packet[0].qubit_vertex]:
            if (
                packet.packet_index > initial_index
                and packet.packet_index < final_index
            ):
                embedded_packets.append(packet)

        return embedded_packets

    def get_all_embedded_packets_for_qubit_vertex(
        self, qubit_vertex
    ) -> Dict[Tuple[Packet, ...], List[Packet]]:
        embedded_packets = {}
        for hopping_packet in self.hopping_packets[qubit_vertex]:
            embedded_packets[hopping_packet] = self.get_embedded_packets(
                hopping_packet
            )
        return embedded_packets

    def get_all_embedded_packets(self):
        embedded_packets = {}
        for qubit_vertex in self.hypergraph_circuit.get_qubit_vertices():
            embedded_packets_for_qubit_vertex = (
                self.get_all_embedded_packets_for_qubit_vertex(qubit_vertex)
            )
            if embedded_packets_for_qubit_vertex:
                embedded_packets[
                    qubit_vertex
                ] = embedded_packets_for_qubit_vertex
        return embedded_packets

    def get_nx_graph_conflict(self):
        graph = nx.Graph()
        conflict_edges = []
        checked_hopping_packets = []
        bipartitions = {
            0: [],  # Bottom half
            1: [],  # Top half
        }
        # Iterate through each packet that can be embedded in a hopping packet
        for qubit_vertex in self.hypergraph_circuit.get_qubit_vertices():
            for (
                embedded_packets
            ) in self.get_all_embedded_packets_for_qubit_vertex(
                qubit_vertex
            ).values():
                for embedded_packet in embedded_packets:
                    # Find the connected packet(s) to the embedded packet
                    connected_packets = self.get_connected_packets(
                        embedded_packet
                    )
                    for connected_packet in connected_packets:
                        # If this connected packet has already
                        # been dealt with we can skip it
                        if connected_packet in checked_hopping_packets:
                            continue

                        # Only care if the connected packet is also embedded
                        if self.is_packet_embedded(connected_packet):
                            conflict_edges.append(
                                self.get_conflict_edge(
                                    embedded_packet, connected_packet
                                )
                            )

                            bipartitions = self.assign_to_bipartitions(
                                embedded_packet, connected_packet, bipartitions
                            )

                        checked_hopping_packets.append(embedded_packet)
        logging.debug(f"Conflict edges: {conflict_edges}")
        graph.add_edges_from(conflict_edges)
        return graph, bipartitions[1]

    def get_hopping_packet_from_embedded_packet(self, embedded_packet: Packet):
        """Given a packet that can be embedded,
        return the hopping packet in which it is embedded
        """
        for hopping_packet in self.hopping_packets[
            embedded_packet.qubit_vertex
        ]:
            if (
                hopping_packet[0].packet_index < embedded_packet.packet_index
                and hopping_packet[1].packet_index
                > embedded_packet.packet_index
            ):
                return hopping_packet

    def is_packet_embedded(self, packet: Packet) -> bool:
        return (
            packet.qubit_vertex in self.get_all_embedded_packets().keys()
            and any(
                packet in value
                for value in self.get_all_embedded_packets_for_qubit_vertex(
                    packet.qubit_vertex
                ).values()
            )
        )

    def get_conflict_edge(
        self, embedded_packet1: Packet, embedded_packet2: Packet
    ) -> Tuple[Tuple[Packet, ...], Tuple[Packet, ...]]:
        """This is a very specific function to replace
        long lines of code in `get_nx_graph_conflict()`
        """
        return (
            self.get_hopping_packet_from_embedded_packet(embedded_packet1),
            self.get_hopping_packet_from_embedded_packet(embedded_packet2),
        )

    def assign_to_bipartitions(
        self,
        embedded_packet: Packet,
        connected_packet: Packet,
        bipartitions: Dict[int, List[Tuple[Packet, ...]]],
    ) -> Dict[int, List[Tuple[Packet, ...]]]:
        """This is a very specific function to replace
        long lines of code in `get_nx_graph_conflict()`
        """
        if (
            self.get_hopping_packet_from_embedded_packet(embedded_packet)
            in bipartitions[0]
        ):
            is_top_half = False
        elif (
            self.get_hopping_packet_from_embedded_packet(embedded_packet)
            in bipartitions[1]
        ):
            is_top_half = True
        else:
            is_top_half = True
            bipartitions[is_top_half].append(
                self.get_hopping_packet_from_embedded_packet(connected_packet)
            )

        bipartitions[not is_top_half].append(
            self.get_hopping_packet_from_embedded_packet(connected_packet)
        )
        return bipartitions
