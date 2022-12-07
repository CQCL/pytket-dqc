from __future__ import annotations  # May be redundant in future (PEP 649)
import logging
from numpy import isclose
import networkx as nx  # type: ignore
from networkx.algorithms import bipartite  # type: ignore
from typing import (
    Optional,
    NamedTuple,
)

from pytket_dqc.circuits import HypergraphCircuit, Hyperedge, Vertex
from pytket_dqc.placement import Placement
from pytket.circuit import Command, OpType, Op  # type: ignore
from pytket_dqc.utils import (
    is_distributable,
    to_euler_with_two_hadamards,
)

logger = logging.getLogger()
logger.setLevel(logging.INFO)


class Packet(NamedTuple):
    """The basic structure of a collection of
    distributable gates for determining distributable gates
    whose starting and ending processes.

    These are similar to ``Hyperedges`` in that they contain:
        - A single qubit vertex from the ``HypergraphCircuit``
        - Gate vertices corresponding to CU1 gates in the ``HypergraphCircuit``

    They differ in that they can only contain gate vertices
    whose connected qubits are assigned to the same server.

    On a given qubit vertex, ``Packet``s can be ordered by their
    ``packet_index``s which weakly describe their relation to
    each other on the circuit.
    A ``Packet`` with a lower index means that the first gate in the ``Packet``
    before the first gate of a ``Packet`` with a higher index.
    If the ``Packet``s have the same connected server, then all of the gates
    in the ``Packet`` with the lower index occur before the ``Packet``
    with the higher index.

    :param packet_index: The index by which this packet is labelled.
    :type packet_index: int
    :param qubit_vertex: The qubit vertex shared by all
    the gates in this ``Packet``.
    :type qubit_vertex: Vertex
    :param connected_server_index: The index of the server that all
    the gates are connected to.
    :type connected_server_index: int
    :param gate_vertices: A list of gate vertices.
    :type gate_vertices: list[Vertex]
    :param parent_hedge: The `Hyperedge` from which this
    packet is originally made.
    :type parent_hedge: `Hyperedge`
    """

    packet_index: int
    qubit_vertex: Vertex
    connected_server_index: int
    gate_vertices: list[Vertex]
    parent_hedge: Hyperedge

    def __str__(self):
        return f"P{self.packet_index}"

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return hash(repr(self))


MergedPacket = tuple[Packet, ...]
NeighbouringPacket = tuple[Packet, ...]
HoppingPacket = tuple[Packet, Packet]


class PacMan:
    """Pac(ket)Man(ager)
    Creates and manages ``Packet``s from a given ``HypergraphCircuit``
    and ``Placement``, as well as identifying ``Packet``s that can be
    merged by 'neighbouring' packing and 'hopping' packing.

    NOTE: A hopping packet is only comprised of two ``Packet``s, neighbouring
    packets are of any length >= 2 and merged packets can be of
    length >= 1.

    :param hypergraph_circuit: The ``HypergraphCircuit`` from which
    to build and identify ``Packet``s.
    :type hypergraph_circuit: HypergraphCircuit
    :param placement: The ``Placement`` describing the placement of qubits
    and their gates.
    :type placement: Placement
    :param packets_by_qubit: A dictionary containing chronologically
    ordered lists of all the ``Packet``s on each qubit, with
    each qubit's ``Vertex`` as the key.
    :type packets_by_qubit: dict[Vertex, list[Packet]]
    :param neighbouring_packets: A dictionary containing lists of ``Packet``
    tuples on each qubit. A tuple of ``Packet``s represents a group of packets
    that can be packed together via neighbouring packing.
    :type neighbouring_packets: dict[Vertex, list[NeighbouringPacket]]
    :param hopping_packets: A dictionary containing lists of ``Packet`` tuples
    than can be merged by hopping packing on each qubit.
    A tuple of ``Packet``s represents two packets that can be packed together
    via hopping packing.
    :type hopping_packets: dict[Vertex, list[HoppingPacket]]
    :param merged_packets: A dictionary containing lists of ``Packet`` tuples
    than can be merged by neighbouring or hopping packing on each qubit.
    A tuple of ``Packet``s represents two packets that can be packed together,
    either by neighbouring or hopping packing.
    """

    def __init__(
        self, hypergraph_circuit: HypergraphCircuit, placement: Placement
    ):
        self.hypergraph_circuit: HypergraphCircuit = hypergraph_circuit
        self.placement: Placement = placement
        self.packets_by_qubit: dict[Vertex, list[Packet]] = dict()
        self.neighbouring_packets: dict[            Vertex, list[NeighbouringPacket]
        ] = dict()
        self.hopping_packets: dict[Vertex, list[HoppingPacket]] = dict()
        self.merged_packets: dict[Vertex, list[MergedPacket]] = dict()
        self.build_packets()
        self.identify_neighbouring_packets()
        self.identify_hopping_packets()
        self.merge_all_packets()

    # The basic methods that are called in __init__()

    def build_packets(self):
        """Populate ``.packets_by_qubit``
        by creating ``Packet``s from ``Hyperedge``s.
        Essentially split them up if they go to different servers.
        """
        # The fact they are ordered is guarateed
        # by a predicate on ``HypergraphCircuit``
        hyperedges_ordered = self.hypergraph_circuit.hyperedge_dict
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
        """Populate ``.neighbouring_packets`` by finding groups of packets
        that can be merged by neighbouring packing.

        Groups of packets that can be merged by neighbouring packing are
        placed together in a tuple.
        """
        logger.debug("------------------------------")
        logger.debug("Identifying neighbouring packets.")
        for qubit_vertex in self.hypergraph_circuit.get_qubit_vertices():
            logger.debug(
                f"Checking qubit vertex {qubit_vertex} for nghbrng packets."
            )
            neighbouring_packets: list[NeighbouringPacket] = list()
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
        """Populate ``.hopping_packets`` by finding groups of packets that
         can be merged by hopping packing

        Groups of packets that can be merged by hopping packing are
        placed together in a tuple.
        """
        logger.debug("------------------------------")
        logger.debug("Identifying hopping packets.")
        for qubit_vertex in self.hypergraph_circuit.get_qubit_vertices():
            logger.debug(
                f"Checking qubit vertex {qubit_vertex} for hopping packets."
            )
            self.hopping_packets[qubit_vertex] = []
            for packet in self.packets_by_qubit[qubit_vertex][:-1]:
                packet_to_compare = self.get_next_packet(
                    self.get_next_packet(packet)
                )
                while packet_to_compare is not None:
                    logger.debug(f"Comparing {packet} and {packet_to_compare}")
                    if self.are_hoppable_packets(packet, packet_to_compare):
                        self.hopping_packets[qubit_vertex].append(
                            (packet, packet_to_compare)
                        )
                        break
                    packet_to_compare = self.get_next_packet(packet_to_compare)

    def merge_all_packets(self):
        """Populate ``.merged_packets`` by merging together already identified
        neighbouring and hopping packets.

        Groups of packets that are merged are placed together in a tuple.
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
                            f"Adding {packet} to (nghbrng) {mergeable_packets}"
                        )
                        mergeable_packets.append(packet)
                        considered_packets.append(packet)
                    elif (
                        self.get_subsequent_hopping_packet(packet) is not None
                    ):
                        packet = self.get_subsequent_hopping_packet(packet)
                        logger.debug(
                            f"Adding {packet} to (hopping) {mergeable_packets}"
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
        """Given a ``Packet``, find the next ``Packet`` that can be
        packed with the given ``Packet`` by neighbouring packing.

        Returns ``None`` if no such ``Packet`` exists.

        :param packet: The ``Packet`` for which the subsequent
        neighbouring ``Packet`` should be found.
        :type packet: Packet
        :return: The subsequent neighbouring ``Packet``.
        :rtype: Optional[Packet]
        """
        potential_neighbouring_packets = [
            neighbouring_packets
            for neighbouring_packets in self.neighbouring_packets[
                packet.qubit_vertex
            ]
            if packet in neighbouring_packets
        ]
        assert (
            len(potential_neighbouring_packets) <= 1
        ), "Can only be 1 neighbouring packet containing the given packet."
        if len(potential_neighbouring_packets) == 0:
            return None
        idx = potential_neighbouring_packets[0].index(packet)
        if idx + 1 == len(potential_neighbouring_packets[0]):
            return None

        return potential_neighbouring_packets[0][idx + 1]

    def get_subsequent_hopping_packet(
        self, packet: Packet
    ) -> Optional[Packet]:
        """Given a ``Packet``, find the next ``Packet`` that can be packed
        with the given ``Packet`` by hopping packing.

        Returns ``None`` if no such ``Packet`` exists.

        :param packet: The ``Packet`` for which the subsequent hopping
        ``Packet`` should be found.
        :type packet: Packet
        :return: The subsequent hopping ``Packet``.
        :rtype: Optional[Packet]
        """
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
        assert (
            len(potential_hopping_packets[0]) == 2
        ), "Hopping packets are always length two."
        return potential_hopping_packets[0][1]

    def get_next_packet(self, packet: Optional[Packet]) -> Optional[Packet]:
        """Given a ``Packet``, find the next ``Packet`` on the same qubit
        with the same connected server.

        Returns ``None`` if no such ``Packet`` exists, or if ``None``
        is passed in.

        :param packet: The ``Packet`` for which the subsequent hopping
        ``Packet`` should be found.
        :type packet: Optional[Packet]
        :return: The subsequent ``Packet``.
        :rtype: Optional[Packet]
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

    def get_all_packets(self) -> list[Packet]:
        """Return a list of all ``Packet``s on the entire circuit,
        sorted in index order.

        NOTE: Indices only describe chronological order of ``Packet``s on the
        same qubit vertex, so there is no gurantee in general that
        ``all_packets[i]`` is earlier than ``all_packets[i + 1]``.

        :return: All the packets on the entire circuit.
        :rtype: List[Packet]
        """
        all_packets: list[Packet] = []
        for packets in self.packets_by_qubit.values():
            all_packets += packets
        all_packets.sort(key=lambda x: x.packet_index)
        return all_packets

    def get_all_merged_packets(self) -> list[MergedPacket]:
        merged_packets: list[MergedPacket] = list()
        for i in range(len(self.merged_packets.keys())):
            merged_packets.extend(self.merged_packets[i])
        return merged_packets

    def get_connected_server(
        self, qubit_vertex: Vertex, gate_vertex: Vertex
    ) -> int:
        """Given a qubit and gate ``Vertex`` corresponding to a CU1 gate,
        find the server number of the server containing the
        other qubit of the gate.

        :param qubit_vertex: The qubit ``Vertex``
        which we are not concerned about the server index.
        :type qubit_vertex: Vertex
        :param gate_vertex: The CU1 gate of interest.
        :type gate_vertex: Vertex
        :return: The server index of the other server.
        :rtype: int
        """

        assert self.hypergraph_circuit.is_qubit_vertex(qubit_vertex)
        assert not self.hypergraph_circuit.is_qubit_vertex(gate_vertex)
        command_index = (
            self.hypergraph_circuit.get_vertex_to_command_index_map()[
                gate_vertex
            ]
        )
        command_dict = self.hypergraph_circuit._commands[command_index]
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
    ) -> bool:
        """Given two packets, determine if they can
        be packed by neighbouring packing.

        :param first_packet: The first packet of interest.
        :type first_packet: Packet
        :param second_packet: The second packet of interest.
        :type second_packet: Packet
        :return: Whether the packets can be packed by neighbouring packing.
        :rtype: bool
        """

        if (
            first_packet.qubit_vertex != second_packet.qubit_vertex
            or first_packet.connected_server_index
            != second_packet.connected_server_index
        ):
            return False

        intermediate_commands = self.get_intermediate_commands(
            first_packet, second_packet
        )
        # Since DQCPass() removes X gates,
        # need to see if there is a sequence
        # HZH (= X) in the intermediate commands
        # The below switches activate accordingly when
        # - An initial H is found (searching_for_Z is True)
        # - Next command is a Z (searching_for_Z False, searching_for_H True)
        # - Next command is a H (searching_for_H is False)
        # Then we can continue as usual
        searching_for_Z = False
        searching_for_H = False
        for command in intermediate_commands:
            if searching_for_Z:
                if (
                    command.op.type == OpType.Rz
                    and abs(command.op.params[0]) == 1
                ):
                    logger.debug("Found Z, looking for second H")
                    searching_for_Z = False
                    searching_for_H = True
                else:
                    return False
            elif searching_for_H:
                if command.op.type == OpType.H:
                    logger.debug("Found second H, got an X overall")
                    searching_for_H = False
                else:
                    return False
            elif command.op.type == OpType.H:
                logger.debug("Found first H, now looking for Z")
                searching_for_Z = True
            elif not is_distributable(command.op):
                return False

        return not (searching_for_H or searching_for_Z)

    def are_hoppable_packets(
        self, first_packet: Packet, second_packet: Packet
    ) -> bool:
        """Given two packets, determine if they can
        be packed by hopping packing.

        :param first_packet: The first packet of interest.
        :type first_packet: Packet
        :param second_packet: The second packet of interest.
        :type second_packet: Packet
        :return: Whether the packets can be packed by hopping packing.
        :rtype: bool
        """
        assert first_packet.qubit_vertex == second_packet.qubit_vertex

        logger.debug(
            f"Are packets {first_packet} and "
            + f"{second_packet} packable via embedding?"
        )

        intermediate_commands = self.get_intermediate_commands(
            first_packet, second_packet
        )
        logger.debug(
            f"Int ops {[command.op for command in intermediate_commands]}"
        )

        # Find and check that all the CU1 gates are embeddable
        # Store their indices in ``intermediate_commands`` so that
        # it can be sliced later into lists of 1 qubit gates between
        # the CU1 gates.
        cu1_indices = []
        for i, command in enumerate(intermediate_commands):
            if command.op.type == OpType.CU1:
                assert (
                    first_packet.connected_server_index
                    == second_packet.connected_server_index
                ), "Cannot pack packets connected to different servers"
                if not self.hypergraph_circuit.is_h_embeddable_CU1(
                    command,
                    set(
                        [
                            self.placement.placement[
                                first_packet.qubit_vertex
                            ],
                            first_packet.connected_server_index,
                        ]
                    ),
                    self.placement,
                ):
                    logger.debug(
                        f"No, CU1 command {command} is not embeddable."
                    )
                    return False
                cu1_indices.append(i)

        assert cu1_indices, "There must be CU1 gates between the two packets."

        # Add the ops at the start of the embedding
        ops_1q_list: list[list[Op]] = [
            [
                command.op
                for command in intermediate_commands[0: cu1_indices[0]]
            ]
        ]

        # Convert the intermediate commands between CU1s
        # to ensure they have 2 Hadamards
        # barring the initial set of commands and the final
        # set of commands
        prev_cu1_index = cu1_indices[0]
        for cu1_index in cu1_indices[1:]:
            commands = intermediate_commands[prev_cu1_index + 1: cu1_index]
            ops = [command.op for command in commands]
            ops_1q_list.append(to_euler_with_two_hadamards(ops))
            prev_cu1_index = cu1_index

        # Add the ops at the end of the embedding
        ops_1q_list.append(
            [
                command.op
                for command in intermediate_commands[cu1_indices[-1] + 1:]
            ]
        )

        logger.debug(f"ops_1q_list {ops_1q_list}")

        n_hadamards_start = len(
            [op for op in ops_1q_list[0] if op.type == OpType.H]
        )
        if n_hadamards_start != 1:
            logger.debug(
                "No, there must be 1 Hadamard "
                + "at the start of the Hadamard sandwich."
            )
            return False

        n_hadamards_end = len(
            [op for op in ops_1q_list[-1] if op.type == OpType.H]
        )
        if n_hadamards_end != 1:
            logger.debug(
                "No, there must be 1 Hadamard "
                + "at the end of the Hadamard sandwich."
            )
            return False

        # In the case that two Hadamards were inserted at the end of
        # just a single Rz gate, we need to check if the embedding
        # condition is met if we insert the two Hadamards at the start
        # I.e we have
        # [Rz(x), H, Rz(y), H, Rz(z)], [Rz(a), H, Rz(0), H, Rz(0)]
        # and the condition is not met then must also check
        # [Rz(x), H, Rz(y), H, Rz(z)], [Rz(0), H, Rz(0), H, Rz(a)]
        # If the condition is met in the second instance then we
        # record that the list must be reversed when considering with the
        # next set of gates.
        is_reversed = False
        for first_ops_1q, second_ops_1q in zip(
            ops_1q_list[0:-1], ops_1q_list[1:]
        ):
            if is_reversed:
                first_ops_1q_to_try = list(reversed(first_ops_1q))
            else:
                first_ops_1q_to_try = first_ops_1q
            is_reversed = False
            if not (
                self.are_1q_op_phases_npi(first_ops_1q_to_try, second_ops_1q)
            ):
                if (
                    len(second_ops_1q) == 5
                    and second_ops_1q[2].params == 0
                    and second_ops_1q[4].params == 0
                ):
                    if self.are_1q_op_phases_npi(
                        first_ops_1q_to_try, list(reversed(second_ops_1q))
                    ):
                        is_reversed = True
                    else:
                        logger.debug(
                            f"No, the phases of {first_ops_1q} and "
                            + f"{second_ops_1q} prevent embedding."
                        )
                    return False
                else:
                    return False

        logger.debug("YES!")
        return True

    def get_conflict_hoppings(
        self, hopping_packet: HoppingPacket
    ) -> list[HoppingPacket]:
        """Given a `HoppingPacket`, determine the other `HoppingPacket`s with
        which it conflicts.

        i.e. see if any of the `Packet`s embedded in it are connnected
        to other `Packet`s which are also embedded.
        If so return the `HoppingPacket`s that contain them.

        :param hopping_packet: The `HoppingPacket` to check
        :type hopping_packet: HoppingPacket
        :return: Return the other `HoppingPacket`s that form a conflict with it
        :rtype: list[HoppingPacket]
        """
        conflict_hoppings: list[HoppingPacket] = []

        for embedded_packet in self.get_embedded_packets(hopping_packet):
            for connected_packet in self.get_connected_packets(
                embedded_packet
            ):
                if self.is_packet_embedded(connected_packet):
                    conflict_hoppings.append(
                        self.get_hopping_packet_from_embedded_packet(
                            connected_packet
                        )
                    )
        return conflict_hoppings

    # Methods that interface between Packets and HypergraphCircuit

    def hyperedge_to_packets(
        self, hyperedge: Hyperedge, current_index: int
    ) -> tuple[int, list[Packet]]:
        """Given a ``Hyperedge``, convert it into a list of ``Packet``s,
        labelling with the appropriate index.

        :param hyperedge: The ``Hyperedge`` to convert.
        :type hyperedge: Hyperedge
        :param current_index: The index to start labelling ``Packet``s from
        :type current_index: int
        :return: The index from which to carry on labelling future
        ``Packet``s and the list of created ``Packet``s.
        :rtype: tuple[int, list[Packet]]
        """
        hyperedge_qubit_vertex = self.hypergraph_circuit.get_qubit_vertex(
            hyperedge
        )
        connected_server_to_dist_gates: dict[
            int, list[Vertex]
        ] = {}  # Server number to list of distributed gates on that server.
        packets: list[Packet] = []
        for gate_vertex in self.hypergraph_circuit.get_gate_vertices(
            hyperedge
        ):
            connected_server = self.get_connected_server(
                hyperedge_qubit_vertex, gate_vertex
            )
            if (
                connected_server
                == self.placement.placement[hyperedge_qubit_vertex]
            ):
                continue
            elif connected_server in connected_server_to_dist_gates.keys():
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
                    current_index,
                    hyperedge_qubit_vertex,
                    connected_server,
                    dist_gates,
                    hyperedge,
                )
            )
            current_index += 1

        return current_index, packets

    def get_intermediate_commands(
        self, first_packet: Packet, second_packet: Packet
    ) -> list[Command]:
        """Given two packets, find all the commands between
        the last gate of the first packet
        and the first gate of the last packet.

        :param first_packet: The first ``Packet`` of interest.
        :type first_packet: Packet
        :param second_packet: The second ``Packet`` of interest.
        :type second_packet: Packet
        :return: The commands between the two ``Packet``s
        :rtype: list[Command]
        """
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
        self, prior_1q_ops: list[Op], post_1q_ops: list[Op]
    ) -> bool:
        """Given two lists of ``Op``s, determine whether these allow
        for the CU1 that these operations sandwich to be embedded,
        by calculating if the relevant phases
        either side of the CU1 sum to an integer.

        :param prior_1q_ops: List of ``Op``s prior to a CU1.
        :type prior_1q_ops: list[Op]
        :param post_1q_ops: List of ``Op``s after a CU1.
        :type post_1q_ops: list[Op]
        :return: Whether the phases sum to an integer.
        :rtype: bool
        """

        # If the length of prior ops is not 5,
        # then the list has not been passed into
        # ``to_euler_with_two_hadamards``
        # This only happens for the very first
        # set of ops in the embedding, which is the
        # only time we may not have an Rz at the end
        # of the ops list
        if not len(prior_1q_ops) == 5 and prior_1q_ops[-1].type == OpType.H:
            prior_phase = 0

        else:
            prior_op = prior_1q_ops[-1]
            assert prior_op.type == OpType.Rz
            prior_phase = prior_op.params[0]

        # If the length of post ops is not 5,
        # then the list has not been passed into
        # ``to_euler_with_two_hadamards``
        # This only happens for the very last
        # set of ops in the embedding, which is the
        # only time we may not have an Rz at the start
        # of the ops list.
        if not len(post_1q_ops) == 5 and post_1q_ops[0].type == OpType.H:
            post_phase = 0

        else:
            post_op = post_1q_ops[0]
            assert post_op.type == OpType.Rz
            post_phase = post_op.params[0]

        phase_sum = prior_phase + post_phase
        return bool(isclose(phase_sum % 1, 0) or isclose(phase_sum % 1, 1))

    def get_connected_packets(self, packet: Packet) -> set[Packet]:
        """Get all the ``Packet``s connected to the
        gate vertices in this ``Packet``.

        :param packet: The ``Packet`` to find connections to.
        :type packet: Packet
        :return: The connected ``Packet``s.
        :rtype: set[Packet]
        """
        connected_packets = set()
        for gate_vertex in packet.gate_vertices:
            gate_qubits = self.hypergraph_circuit.get_gate_of_vertex(
                gate_vertex
            ).qubits
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
                        connected_packets.add(potential_packet)
                    break
        return connected_packets

    def get_containing_merged_packet(self, packet: Packet) -> MergedPacket:
        """Given a ``Packet`` return the merged packet containing it.

        :param packet: The ``Packet`` of interest.
        :type packet: Packet
        :return: The containing merged packet.
        :rtype: MergedPacket
        """
        for merged_packet in self.merged_packets[packet.qubit_vertex]:
            if packet in merged_packet:
                containing_packet = merged_packet
                break
        return containing_packet

    def get_connected_merged_packets(
        self, merged_packet: MergedPacket
    ) -> set[MergedPacket]:
        """Given a merged packet, find all its connected merged packets.

        :param merged_packet: The merged packet of interest
        :type merged_packet: MergedPacket
        :return: The set of connected merged packets.
        :rtype: set[MergedPacket]
        """
        connected_merged_packets: set[MergedPacket] = set()
        for packet in merged_packet:
            connected_merged_packets.update(
                [
                    self.get_containing_merged_packet(connected_packet)
                    for connected_packet in self.get_connected_packets(packet)
                ]
            )

        return connected_merged_packets

    def get_embedded_packets(
        self, hopping_packet: HoppingPacket
    ) -> set[Packet]:
        """For a given hopping packet,
        find all the ``Packet``s embedded inside it.

        :param hopping_packet: The hopping packet of interest.
        :type hopping_packet: HoppingPacket
        :return: Set of embedded packets.
        :rtype: set[Packet]
        """
        initial_index = hopping_packet[0].packet_index
        final_index = hopping_packet[1].packet_index
        assert initial_index < final_index
        embedded_packets: set[Packet] = set()
        for packet in self.packets_by_qubit[hopping_packet[0].qubit_vertex]:
            if (
                packet.packet_index > initial_index
                and packet.packet_index < final_index
            ):
                embedded_packets.add(packet)

        return embedded_packets

    def get_all_embedded_packets_for_qubit_vertex(
        self, qubit_vertex: Vertex
    ) -> dict[HoppingPacket, set[Packet]]:
        """Get all the embedded packets on a given qubit ``Vertex``,
        with keys being the hopping packets that embed them.

        :param qubit_vertex: Qubit ``Vertex`` of interest.
        :type qubit_vertex: Vertex
        :return: Dictionary of sets of embedded packets,
        keyed by their embedding hopping packets.
        :rtype: dict[HoppingPacket, set[Packet]]
        """
        embedded_packets: dict = {}
        for hopping_packet in self.hopping_packets[qubit_vertex]:
            embedded_packets[hopping_packet] = self.get_embedded_packets(
                hopping_packet
            )
        return embedded_packets

    def get_all_embedded_packets(
        self,
    ) -> dict[Vertex, dict[HoppingPacket, set[Packet]]]:
        """Get all the embedded packets.

        Nested dictionary keys follow:
        qubit ``Vertex`` -> hopping packet -> set of embedded ``Packet``s.

        :return: A nested dictionary leading to sets
        of all embedded ``Packet``s.
        :rtype: dict[Vertex, dict[HoppingPacket, set[Packet]]]
        """
        embedded_packets: dict = {}
        for qubit_vertex in self.hypergraph_circuit.get_qubit_vertices():
            embedded_packets_for_qubit_vertex = (
                self.get_all_embedded_packets_for_qubit_vertex(qubit_vertex)
            )
            if embedded_packets_for_qubit_vertex:
                embedded_packets[
                    qubit_vertex
                ] = embedded_packets_for_qubit_vertex
        return embedded_packets

    def get_hopping_packet_from_embedded_packet(
        self, embedded_packet: Packet
    ) -> HoppingPacket:
        """Given a ``Packet`` that is embedded in a hopping packet,
        find that hopping packet.

        :param embedded_packet: The embedded ``Packet``.
        :type embedded_packet: Packet
        :return: The hopping packet that embeds it.
        :rtype: HoppingPacket
        """
        for hopping_packet in self.hopping_packets[
            embedded_packet.qubit_vertex
        ]:
            if (
                hopping_packet[0].packet_index < embedded_packet.packet_index
                and hopping_packet[1].packet_index
                > embedded_packet.packet_index
            ):
                break
        return hopping_packet

    def is_packet_embedded(self, packet: Packet) -> bool:
        """Checks if a ``Packet`` is embedded

        :param packet: The ``Packet`` of interest.
        :type packet: Packet
        :return: Whether the ``Packet`` is embedded.
        :rtype: bool
        """
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
                for (
                    connected_merged_packet
                ) in self.get_connected_merged_packets(merged_packet):
                    edges.add((merged_packet, connected_merged_packet))

        graph.add_edges_from(edges)
        bipartitions = self.assign_bipartitions(graph)
        assert self.is_bipartite_predicate(graph, edges, bipartitions)
        return graph, bipartitions[1]

    def get_nx_graph_neighbouring(self):
        """Get the NetworkX graph representing
        the circuit assuming only neighbouring packing.
        Nodes are neighbouring packets.
        """
        graph = nx.Graph()
        edges = set()

        for packet in self.get_all_packets():
            for connected_packet in self.get_connected_packets(packet):
                if (packet, connected_packet) not in edges:
                    edges.add((packet, connected_packet))

        for qubit_vertex in self.hypergraph_circuit.get_qubit_vertices():
            for neighbouring_packet in self.neighbouring_packets[qubit_vertex]:
                for packet in neighbouring_packet:
                    relevant_edges = [edge for edge in edges if packet in edge]
                    for edge in relevant_edges:
                        assert len(edge - (packet)) == 1
                        (other_node,) = edge - (packet)
                        edges.remove(edge)
                        edges.add((neighbouring_packet, other_node))

        graph.add_edges_from(edges)
        bipartitions = self.assign_bipartitions(graph)
        assert self.is_bipartite_predicate(graph, edges, bipartitions)
        return graph, bipartitions[1]

    def get_nx_graph_conflict(self):
        """Get the NetworkX graph representing conflict edges.
        Nodes are hopping packets that represent conflicts.
        """
        graph = nx.Graph()
        potential_conflict_edges = set()
        checked_hopping_packets = []

        # Iterate through each packet that can be embedded in a hopping packet
        for qubit_vertex in self.hypergraph_circuit.get_qubit_vertices():
            for hopping_packet in self.hopping_packets[qubit_vertex]:
                for conflict_hopping in self.get_conflict_hoppings(
                    hopping_packet
                ):
                    if conflict_hopping in checked_hopping_packets:
                        continue
                    potential_conflict_edges.add(
                        (hopping_packet, conflict_hopping)
                    )
                checked_hopping_packets.append(hopping_packet)
        graph.add_edges_from(potential_conflict_edges)
        bipartitions = self.assign_bipartitions(graph)
        assert self.is_bipartite_predicate(
            graph, potential_conflict_edges, bipartitions
        )
        return graph, bipartitions[1]

    def get_mvc_merged_graph(self) -> set[MergedPacket]:
        """Get the minimum vertex cover of the merged graph."""
        g, topnodes = self.get_nx_graph_merged()
        matching = bipartite.maximum_matching(g, top_nodes=topnodes)
        return bipartite.to_vertex_cover(g, matching, top_nodes=topnodes)

    def get_mvc_neighbouring_graph(self) -> set[NeighbouringPacket]:
        """Get the minimum vertex cover of the neighbouring graph."""
        g, topnodes = self.get_nx_graph_neighbouring()
        matching = bipartite.maximum_matching(g, top_nodes=topnodes)
        return bipartite.to_vertex_cover(g, matching, top_nodes=topnodes)

    def get_conflict_edges_given_mvc(
        self,
        potential_conflict_edges: set[frozenset[HoppingPacket]],
        mvc: set[MergedPacket],
    ) -> set[tuple[HoppingPacket, HoppingPacket]]:
        """Given an MVC, find all the edges in the
        conflict graph that represent true conflicts.

        True conflicts means that both nodes of the edge are in the MVC.

        :param potential_conflict_edges: The set of all edges that would be
        conflicting if both nodes were to be in an MVC.
        :type potential_conflict_edges: set[frozenset[HoppingPacket]]
        :param mvc: The minimum vertex cover to use to find true conflicts.
        :type mvc: set[MergedPacket]
        :return: Set of true conflict edges.
        :rtype: set[MergedPacket]
        """
        true_conflicts = set()
        for u, v in potential_conflict_edges:
            assert self.get_containing_merged_packet(
                u[0]
            ) == self.get_containing_merged_packet(u[1])
            assert self.get_containing_merged_packet(
                v[0]
            ) == self.get_containing_merged_packet(v[1])
            if (
                self.get_containing_merged_packet(u[0]) in mvc
                and self.get_containing_merged_packet(v[0]) in mvc
            ):
                true_conflicts.add((u, v))

        return true_conflicts

    def get_conflict_edge(
        self, embedded_packet1: Packet, embedded_packet2: Packet
    ) -> tuple[HoppingPacket, HoppingPacket]:
        """Given two embedded packets, return a ``frozenset``
        that has the hopping packets that embed the packets as elements.

        This is a very specific function to replace
        long lines of code in `get_nx_graph_conflict()`
        that failed flake8 line length checks.
        """
        return (
            self.get_hopping_packet_from_embedded_packet(embedded_packet1),
            self.get_hopping_packet_from_embedded_packet(embedded_packet2),
        )

    def assign_bipartitions(
        self, graph: nx.Graph
    ) -> dict[int, set[MergedPacket]]:
        """Given a graph, for each connected component designate its nodes
        as top or bottom half
        """
        bipartitions: dict[int, set[MergedPacket]] = {
            0: set(),  # Bottom
            1: set(),  # Top
        }

        for subgraph in [
            graph.subgraph(c) for c in nx.connected_components(graph)
        ]:
            bottom_nodes, top_nodes = bipartite.sets(subgraph)
            bipartitions[0].update(bottom_nodes)
            bipartitions[1].update(top_nodes)

        return bipartitions

    def is_bipartite_predicate(self, graph, edges, bipartitions):
        predicate = nx.is_bipartite(graph)
        for edge in edges:
            (u, v) = edge
            predicate = predicate and (
                (u in bipartitions[0] and v in bipartitions[1])
                or (v in bipartitions[0] and u in bipartitions[1])
            )
        return predicate
