from __future__ import annotations  # May be redundant in future (PEP 649)
import logging
from numpy import isclose, bool_
import networkx as nx  # type: ignore
from networkx.algorithms import bipartite  # type: ignore
from typing import List, Dict, Tuple, Optional, Set, FrozenSet, Union, NamedTuple, cast

from pytket_dqc.circuits import HypergraphCircuit, Hyperedge, Vertex
from pytket_dqc.placement import Placement
from pytket.circuit import Command, OpType, Op, Qubit  # type: ignore
from pytket_dqc.utils import (
    is_distributable,
    distributable_1q_op_types,
    distributable_op_types,
    to_euler_with_two_hadamards
)

logger = logging.getLogger()
logger.setLevel(logging.INFO)

class Packet(NamedTuple):
    """The basic structure of a collection of
    distributable gates.
    """

    packet_index: int
    qubit_vertex: Vertex
    connected_server_index: int
    gate_vertices: list[Vertex]

    def __str__(self):
        return f"P{self.packet_index}"

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return hash(repr(self))


class PacMan:
    """Builds and manages packets using a HypergraphCircuit and a Placement"""

    def __init__(
        self, hypergraph_circuit: HypergraphCircuit, placement: Placement
    ):
        self.hypergraph_circuit = hypergraph_circuit
        self.placement = placement
        self.packets_by_qubit: Dict[Vertex, List[Packet]] = dict()
        self.hopping_packets: Dict[Vertex, List[Tuple[Packet, ...]]] = dict()
        self.neighbouring_packets: Dict[
            Vertex, List[Tuple[Packet, ...]]
        ] = dict()
        self.merged_packets: Dict[Vertex, List[Tuple[Packet, ...]]] = dict()
        self.build_packets()
        self.identify_neighbouring_packets()
        self.identify_hopping_packets()
        self.merge_all_packets()

    # The basic methods that are called in __init__()

    def build_packets(self):
        """Build packets from `Hyperedges`
        Essentially split them up if they go to different servers
        """
        hyperedges_ordered = self.get_hyperedges_ordered()
        current_index = 0
        for qubit_vertex in self.hypergraph_circuit.get_qubit_vertices():
            self.packets_by_qubit[qubit_vertex] = []
            for hyperedge in hyperedges_ordered[qubit_vertex]:
                current_index, packets = self.hyperedge_to_packets(
                    hyperedge, current_index
                )
                for packet in packets:
                    self.packets_by_qubit[qubit_vertex].append(packet)

    def identify_neighbouring_packets(self):
        logger.debug("------------------------------")
        logger.debug("Identifying neighbouring packets.")
        for qubit_vertex in self.hypergraph_circuit.get_qubit_vertices():
            logger.debug(
                f"Checking qubit vertex {qubit_vertex} for neighbouring packets."
            )
            neighbouring_packets: List[Tuple[Packet, ...]] = list()
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
                if len(neighbouring_packet) > 1:
                    neighbouring_packets.append(tuple(neighbouring_packet))
            self.neighbouring_packets[qubit_vertex] = neighbouring_packets

    def identify_hopping_packets(self):
        logger.debug("------------------------------")
        logger.debug("Identifying hopping packets.")
        for qubit_vertex in self.hypergraph_circuit.get_qubit_vertices():
            logger.debug(
                f"Checking qubit vertex {qubit_vertex} for hopping packets."
            )
            self.hopping_packets[qubit_vertex] = []
            for packet in self.packets_by_qubit[qubit_vertex][:-1]:
                packet_to_compare = self.get_next_packet(self.get_next_packet(packet))
                while (
                    packet_to_compare is not None
                ):
                    logger.debug(f"Comparing {packet} and {packet_to_compare}")
                    if self.are_hoppable_packets(packet, packet_to_compare):
                        self.hopping_packets[qubit_vertex].append(
                            (packet, packet_to_compare)
                        )
                        break
                    packet_to_compare = self.get_next_packet(packet_to_compare)

    def merge_all_packets(self):
        """Using the identified neighbouring and hopping
        packets, create tuples merging them together
        """
        logger.debug("------------------------------")
        logger.debug("Merging packets")
        for qubit_vertex in self.hypergraph_circuit.get_qubit_vertices():
            self.merged_packets[qubit_vertex] = []
            considered_packets = []
            for packet in self.packets_by_qubit[qubit_vertex]:
                logger.debug(f"Checking packet {packet}")
                if packet in considered_packets:
                    logger.debug("Already checked")
                    continue
                mergeable_packets = [packet]
                continue_merging = True
                while continue_merging:
                    logger.debug(f"Start of checks {mergeable_packets}")
                    logger.debug(f"Packet of interest is {packet}")
                    if (
                        self.get_subsequent_neighbouring_packet(packet)
                        is not None
                    ):
                        packet = self.get_subsequent_neighbouring_packet(
                            packet
                        )
                        logger.debug(
                            f"Adding {packet} to mergeable packets (neighbouring) {mergeable_packets}"
                        )
                        mergeable_packets.append(packet)
                        considered_packets.append(packet)
                    elif (
                        self.get_subsequent_hopping_packet(packet) is not None
                    ):
                        packet = self.get_subsequent_hopping_packet(packet)
                        logger.debug(
                            f"Adding {packet} to mergeable packets (hopping) {mergeable_packets}"
                        )
                        mergeable_packets.append(packet)
                        considered_packets.append(packet)
                    else:
                        continue_merging = False
                        logger.debug(f"End of checks {mergeable_packets}")

                self.merged_packets[qubit_vertex].append(
                    tuple(mergeable_packets)
                )

    # Utility methods for Packets

    def get_subsequent_neighbouring_packet(
        self, packet: Packet
    ) -> Optional[Packet]:
        potential_neighbouring_packets = [
            neighbouring_packets
            for neighbouring_packets in self.neighbouring_packets[
                packet.qubit_vertex
            ]
            if packet in neighbouring_packets
        ]
        assert len(potential_neighbouring_packets) <= 1,\
            "There can only be 1 neighbouring packet containing the given packet."
        if len(potential_neighbouring_packets) == 0:
            return None
        idx = potential_neighbouring_packets[0].index(packet)
        if idx + 1 == len(potential_neighbouring_packets[0]):
            return None

        return potential_neighbouring_packets[0][idx + 1]

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
        assert len(potential_hopping_packets[0]) == 2,\
            "Hopping packets are always length two."
        return potential_hopping_packets[0][1]

    def get_next_packet(self, packet: Optional[Packet]) -> Optional[Packet]:
        """Find the next packet on the qubit of the given packet that
        is connected to the same server as the given packet.
        Returns None if no such packet exists

        NOTE: None is also returned if None is passed in
        """
        if packet is None:
            return None

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

    def get_connected_server(self, qubit_vertex: Vertex, gate_vertex: Vertex):
        command_index = self.hypergraph_circuit.get_vertex_to_command_index_map()[gate_vertex]
        command_dict = self.hypergraph_circuit._commands[
            command_index
        ]  # TODO: Rewrite this as Pablo intended
        command = command_dict["command"]
        assert (
            type(command) == Command
        )  # This is for mypy check - is there a better way?
        qubits = command.qubits
        qubit_vertex_candidates = [
            self.hypergraph_circuit.get_vertex_of_qubit(qubit)
            for qubit in qubits
            if self.hypergraph_circuit.get_vertex_of_qubit(qubit)
            is not qubit_vertex
        ]

        assert (
            len(qubit_vertex_candidates) == 1
        ), "There should be 1 and only 1 other qubit vertex."
        other_qubit_vertex = qubit_vertex_candidates[0]
        return self.placement.placement[other_qubit_vertex]

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
        assert first_packet.qubit_vertex == second_packet.qubit_vertex

        logger.debug(
            f"Are packets {first_packet} and "
            + f"{second_packet} packable via embedding?"
        )

        allowed_op_types = distributable_op_types + [OpType.H]

        intermediate_commands = self.get_intermediate_commands(
            first_packet, second_packet
        )
        logger.debug(
            f"Intermediate ops {[command.op for command in intermediate_commands]}"
        )
        cu1_indices = []
        for i, command in enumerate(intermediate_commands):
            if command.op.type not in allowed_op_types:
                logger.debug(f"No, {command} is not embeddable.")
                return False
            if command.op.type == OpType.CU1:
                assert first_packet.connected_server_index == second_packet.connected_server_index
                if not self.hypergraph_circuit.is_h_embeddable_CU1(
                    command,
                    set([
                        self.placement.placement[first_packet.qubit_vertex],
                        first_packet.connected_server_index
                    ]),
                    self.placement
                ):
                    logger.debug(
                        f"No, CU1 command {command} is not embeddable."
                    )
                    return False
                cu1_indices.append(i)

        assert cu1_indices

        ops_1q_list: list[list[Op]] = [
            [command.op for command in intermediate_commands[0:cu1_indices[0]]]
        ]

        # Convert the intermediate commands between CU1s
        # barring the initial set of commands and the final
        # set of commands
        prev_cu1_index = cu1_indices[0]
        for cu1_index in cu1_indices[1:]:
            commands = intermediate_commands[
                prev_cu1_index + 1: cu1_index
            ]
            ops = [command.op for command in commands]
            if len([op for op in ops if op.type == OpType.H]) > 0:
                ops_1q_list.append(to_euler_with_two_hadamards(ops))
            else:
                ops_1q_list.append(ops)
            prev_cu1_index = cu1_index
        
        ops_1q_list.append([command.op for command in intermediate_commands[cu1_indices[-1] + 1:]])

        logger.debug(f"ops_1q_list {ops_1q_list}")

        n_hadamards_start = len([op for op in ops_1q_list[0] if op.type == OpType.H])
        if n_hadamards_start != 1:
            logger.debug(
                "No, there can only be 1 Hadamard "
                + "at the start of the Hadamard sandwich."
            )
            return False
        
        n_hadamards_end = len([op for op in ops_1q_list[-1] if op.type == OpType.H])
        if n_hadamards_end != 1:
            logger.debug(
                "No, there can only be 1 Hadamard "
                + "at the end of the Hadamard sandwich."
            )
            return False

        for first_ops_1q, second_ops_1q in zip(ops_1q_list[0:-1], ops_1q_list[1:]):
            if not self.are_1q_op_phases_npi(first_ops_1q, second_ops_1q):
                logger.debug(
                    f"No, the phases of {first_ops_1q} and "
                    + f"{second_ops_1q} prevent embedding."
                )
                return False

        logger.debug("YES!")
        return True

    # Methods that interface between Packets and HypergraphCircuit

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

    def get_intermediate_commands(
        self, first_packet: Packet, second_packet: Packet
    ) -> List[Command]:
        assert (
            first_packet.qubit_vertex == second_packet.qubit_vertex
        ), "Qubit vertices do not match."

        assert first_packet.qubit_vertex == second_packet.qubit_vertex

        qubit_vertex = first_packet.qubit_vertex

        first_vertex = self.hypergraph_circuit.get_last_gate_vertex(
            first_packet.gate_vertices
        )
        second_vertex = self.hypergraph_circuit.get_first_gate_vertex(
            second_packet.gate_vertices
        )

        return self.hypergraph_circuit.get_intermediate_commands(
            first_vertex, second_vertex, qubit_vertex
        )

    def are_1q_op_phases_npi(
        self, prior_1q_ops: List[Op], post_1q_ops: List[Op]
    ) -> bool:
        """Check if the sum of the params of the
        U1 gates that sandwich a CU1 are
        equal to n
        (i.e. the actual phases are equal to n * pi)
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

        return bool(isclose(phase_sum % 1, 0) or isclose(phase_sum % 1, 1))

    def get_connected_packets(self, packet: Packet):
        connected_packets = []
        for gate_vertex in packet.gate_vertices:
            gate_qubits = self.hypergraph_circuit.get_gate_of_vertex(gate_vertex).qubits
            other_qubit_candidates = [
                qubit
                for qubit in gate_qubits
                if self.hypergraph_circuit.get_vertex_of_qubit(qubit)
                != packet.qubit_vertex
            ]
            assert (
                len(other_qubit_candidates) == 1
            ), "There should only be one other qubit candidate \
                for this gate vertex."
            other_qubit_vertex = self.hypergraph_circuit.get_vertex_of_qubit(
                other_qubit_candidates[0]
            )
            for potential_packet in self.packets_by_qubit[other_qubit_vertex]:
                if gate_vertex in potential_packet.gate_vertices:
                    if potential_packet not in connected_packets:
                        connected_packets.append(potential_packet)
                    break
        return connected_packets

    def get_containing_merged_packet(self, packet: Packet):
        for merged_packet in self.merged_packets[packet.qubit_vertex]:
            if packet in merged_packet:
                return merged_packet

    def get_connected_merged_packets(
        self, merged_packet: Tuple[Packet]
    ) -> set[tuple[Packet, ...]]:
        """Also works for neighbouring packets"""
        connected_merged_packets: set[tuple[Packet, ...]] = set()
        for packet in merged_packet:
            connected_packets = self.get_connected_packets(packet)
            for connected_packet in connected_packets:
                connected_merged_packets.add(
                    self.get_containing_merged_packet(connected_packet)
                )

        return connected_merged_packets

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

    # Graph methods

    def get_nx_graph_merged(self):
        """Get the NetworkX graph representing the circuit.
        Nodes are merged packets (neighbouring + hopping).
        """
        graph = nx.Graph()
        edges = set()

        # Add edges to the graph
        for qubit_vertex in self.hypergraph_circuit.get_qubit_vertices():
            for merged_packet in self.merged_packets[qubit_vertex]:
                for connected_merged_packet in self.get_connected_merged_packets(merged_packet):
                    edges.add(frozenset([merged_packet, connected_merged_packet]))

        graph.add_edges_from(edges)
        bipartitions = self.assign_bipartitions(graph)
        self.verify_is_bipartite(graph, edges, bipartitions)
        return graph, bipartitions[1]      

    def get_nx_graph_neighbouring(self):
        """Strategy is to add every edge between packets
        then merge nodes together where the packets
        can be merged by neighbouring packing
        """
        graph = nx.Graph()
        edges = set()

        for packet in self.get_all_packets():
            for connected_packet in self.get_connected_packets(packet):
                if frozenset([packet, connected_packet]) not in edges:
                    edges.add(frozenset([packet, connected_packet]))

        for qubit_vertex in self.hypergraph_circuit.get_qubit_vertices():
            for neighbouring_packet in self.neighbouring_packets[qubit_vertex]:
                for packet in neighbouring_packet:
                    relevant_edges = [edge for edge in edges if packet in edge]
                    for edge in relevant_edges:
                        assert len(edge - frozenset([packet])) == 1
                        (other_node,) = edge - frozenset([packet])
                        edges.remove(edge)
                        edges.add(frozenset([neighbouring_packet, other_node]))
        added_nodes = set()

        graph.add_edges_from(edges)
        bipartitions = self.assign_bipartitions(graph)
        self.verify_is_bipartite(graph, edges, bipartitions)
        return graph, bipartitions[1]

    def get_nx_graph_conflict(self):
        graph = nx.Graph()
        conflict_edges = set()
        checked_hopping_packets = []

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
                            conflict_edges.add(
                                self.get_conflict_edge(
                                    embedded_packet, connected_packet
                                )
                            )
                        checked_hopping_packets.append(embedded_packet)
        logger.debug(f"Conflict edges: {conflict_edges}")
        graph.add_edges_from(conflict_edges)
        bipartitions = self.assign_bipartitions(graph)
        self.verify_is_bipartite(graph, conflict_edges, bipartitions)
        return graph, bipartitions[1]

    def get_mvc_merged_graph(self):
        g, topnodes = self.get_nx_graph_merged()
        matching = bipartite.maximum_matching(g, top_nodes=topnodes)
        return bipartite.to_vertex_cover(g, matching, top_nodes=topnodes)

    def get_mvc_neighbouring_graph(self):
        g, topnodes = self.get_nx_graph_neighbouring()
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

    def get_conflict_edge(
        self, embedded_packet1: Packet, embedded_packet2: Packet
    ) -> FrozenSet[Tuple[Packet, ...]]:
        """This is a very specific function to replace
        long lines of code in `get_nx_graph_conflict()`
        """
        return frozenset(
            [
                self.get_hopping_packet_from_embedded_packet(embedded_packet1),
                self.get_hopping_packet_from_embedded_packet(embedded_packet2),
            ]
        )

    def assign_bipartitions(self, graph: nx.Graph) -> dict[int, set[tuple[Packet, ...]]]:
        """Given a graph, for each connected component designate its nodes
        as top or bottom half
        """
        bipartitions: dict[int, set[tuple[Packet, ...]]] = {
            0: set(),  # Bottom
            1: set(),  # Top
        }

        for subgraph in [graph.subgraph(c) for c in nx.connected_components(graph)]:
            bottom_nodes, top_nodes = bipartite.sets(subgraph)
            bipartitions[0].update(bottom_nodes)
            bipartitions[1].update(top_nodes)
        
        return bipartitions

    def verify_is_bipartite(self, graph, edges, bipartitions):
        assert nx.is_bipartite(graph), "The graph must be bipartite."
        for edge in edges:
            (u, v) = edge
            assert (u in bipartitions[0] and v in bipartitions[1]) or (
                v in bipartitions[0] and u in bipartitions[1]
            )

    # Methods that could be moved to `HypergraphCircuit`

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

    def get_last_command_index(self, gate_vertex_list: List[Vertex]) -> int:
        """Given a list of gate vertices,
        return the index in `Circuit.get_commands()`
        that corresponds to the last gate in the circuit.
        """
        last_command_index: int = -1  # Placeholder value
        for vertex in gate_vertex_list:
            if (
                last_command_index == -1
                or self.hypergraph_circuit.get_vertex_to_command_index_map()[vertex] > last_command_index
            ):
                last_command_index = self.hypergraph_circuit.get_vertex_to_command_index_map()[vertex]

        return last_command_index

    def get_first_command_index(self, gate_vertex_list: List[Vertex]) -> int:
        """Given a list of gate vertices, return the vertex that
        corresponds to the last gate in the circuit.
        """
        first_command_index: int = -1  # Placeholder value
        for vertex in gate_vertex_list:
            if (
                first_command_index == -1
                or self.hypergraph_circuit.get_vertex_to_command_index_map()[vertex] < first_command_index
            ):
                first_command_index = self.hypergraph_circuit.get_vertex_to_command_index_map()[vertex]

        return first_command_index
