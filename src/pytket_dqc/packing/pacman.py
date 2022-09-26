from numpy import allclose, floor, isclose

from pytket_dqc.circuits import HypergraphCircuit, Hyperedge, Vertex
from pytket_dqc.placement import Placement
from pytket.circuit import Command, OpType, Op, Qubit
from pytket_dqc.utils import is_distributable

from typing import List, Dict, Tuple

D = "D"
P = "P"
M = "M"


class Packet():
    """The basic structure of a collection of
    distributable gates.
    """

    def __init__(
        self,
        packet_index: int,
        qubit_vertex: Vertex,
        connected_server_index: int,
        packet_gate_vertices: List[Vertex],
    ):
        self.packet_index: int = packet_index
        self.qubit_vertex: Vertex = qubit_vertex
        self.connected_server_index: int = connected_server_index
        self.packet_gate_vertices: List[Vertex] = packet_gate_vertices
        self.intermediate_commands: List[Command] = []



# class HoppablePacket(Packet):
#     """A collection of CX gates that can be embedded
#     between two other distributable packets.
#     """

#     def __init__(
#         self,
#         packet_index: int,
#         qubit_vertex: Vertex,
#         connected_server_index: int,
#         packet_gate_vertices: List[Vertex],
#         contained_local_gates: Dict[Vertex, List[int]],
#         embeddable_between: List[int],
#     ):
#         super().__init__(
#             self,
#             packet_index,
#             qubit_vertex,
#             connected_server_index,
#             packet_gate_vertices,
#         )
#         self.contained_local_gates = contained_local_gates
#         self.embeddable_between = embeddable_between


class PacMan():
    """Builds and manages packets using a HypergraphCircuit and a Placement
    """

    def __init__(
        self,
        hypergraph_circuit: HypergraphCircuit,
        placement: Placement
    ):
        self.hypergraph_circuit = hypergraph_circuit
        self.placement = placement
        self.packets: List[Packet] = []
        self.packets_by_qubit: Dict[Vertex, List[int]] = {}
        self.vertex_to_command_index: Dict[Vertex, int] = {}
        self.hopping_packets: Dict[Vertex, Dict[int, List[int]]] = {}

        self.build_vertex_to_command_index()
        self.build_packets()
        self.identify_and_merge_neighbouring_packets()
        self.identify_hopping_packets()

    def build_vertex_to_command_index(self):
        """For each vertex in the hypergraph,
        find its corresponding index in the list
        returned by Circuit.get_commands()
        """
        for command_index, command_dict in enumerate(
            self.hypergraph_circuit.commands
        ):
            if command_dict['type'] == 'distributed gate':
                vertex = command_dict['vertex']
                self.vertex_to_command_index[vertex] = command_index

    def build_packets(self):
        # By ordering the hyperedges, their respective packets get put
        # into order by packet_index, up to commutativity of CU1 gates
        hyperedges_ordered = self.get_hyperedges_ordered()
        starting_index = 0
        for qubit_vertex in self.hypergraph_circuit.get_qubit_vertices():
            self.packets_by_qubit[qubit_vertex] = []
            for hyperedge in hyperedges_ordered[qubit_vertex]:
                starting_index, packets = self.hyperedge_to_packets(hyperedge, starting_index)
                self.packets += packets
                for packet in packets:
                    self.packets_by_qubit[qubit_vertex].append(packet.packet_index)

    def identify_and_merge_neighbouring_packets(self):
        for qubit_vertex in self.hypergraph_circuit.get_qubit_vertices():
            to_be_merged: List[List[int]] = []
            already_merged: List[int] = []
            for packet_index in self.packets_by_qubit[qubit_vertex][:-1]:
                if packet_index in already_merged:
                    continue

                next_packet_index = self.get_next_packet_index(packet_index)
                mergeable: List[int] = [packet_index]

                # Find all the packets that can be merged with packet_index
                while next_packet_index is not None:
                    if self.can_be_merged(packet_index, next_packet_index):
                        mergeable.append(next_packet_index)
                        next_packet_index = self.get_next_packet_index(next_packet_index)
                    else:
                        next_packet_index = None
                
                to_be_merged.append(mergeable)
                already_merged += mergeable
            
            for packet_list in to_be_merged:
                if len(packet_list) > 1:
                    self.merge_packets(qubit_vertex, packet_list)

        self.renew_packet_indices()

    def identify_hopping_packets(self):
        # Packets are in order, so I just need to find groups
        # of packets for which the intermediate packets and gates can be embedded

        # YOU ARE DOING THE BELOW!!!!!!!!!
        # Need to change so that if we find a hopping partner, we then see if that hopping partner can have a subsequent embedding
        # Right now if we check beyond first hopping partner it won't work because the hopping partner might not be embeddable
        # OK so the above is done BUT it's quite hacky and doesn't feel very robust

        for qubit_vertex in self.hypergraph_circuit.get_qubit_vertices():
            print(f'Checking qubit vertex {qubit_vertex}')
            self.hopping_packets[qubit_vertex] = []
            checked_packets = []
            for i, packet_index in enumerate(self.packets_by_qubit[qubit_vertex][ : -2]):
                if packet_index in checked_packets:
                    continue
                print(f'Checking packing index {packet_index}')
                print(f'Checking {len(self.packets_by_qubit[qubit_vertex][i + 2 : ])} packets')

                hopping_partners = [packet_index]
                skip_indices = []
                for j in self.packets_by_qubit[qubit_vertex][i + 2 : ]:
                    # Should add a check here that if at any point a CU1 goes to a different
                    # server/is local to stop bothering to check because nothing
                    # can be embedded beyond a local/third server target gate
                    if j in skip_indices:
                        continue

                    print(f'Checking packet {packet_index} against packet {self.packets_by_qubit[qubit_vertex][j]}')
                    
                    hopping_partner_index = self.packets_by_qubit[qubit_vertex][j]

                    if self.are_intermediate_commands_embeddable(packet_index, hopping_partner_index):
                        hopping_partners.append(hopping_partner_index)
                        checked_packets.append(hopping_partner_index)
                        packet_index = hopping_partner_index  # This feels quite dirty
                        skip_indices.append(j+1) # Can't check against the neighbouring packet (otherwise get issues)
                
                if len(hopping_partners) > 1:
                    self.hopping_packets[qubit_vertex].append(hopping_partners)

    def find_hopping_packets(self, packet_index):
        # Find all the packets that can be hopped with this one
        pass

    def renew_packet_indices(self):
        for i, _ in enumerate(self.packets):
            if self.packets[i] is None:
                j = 1
                replacement_packet_found = False
                while i + j < len(self.packets) and not replacement_packet_found:
                    if self.packets[i + j] is not None:
                        not_none_packet = self.packets[i + j]
                        not_none_packet.packet_index = i
                        self.packets_by_qubit[not_none_packet.qubit_vertex].append(i)
                        self.packets_by_qubit[not_none_packet.qubit_vertex].remove(i+j)
                        self.packets[i], self.packets[i + j] = self.packets[i + j], self.packets[i]
                        replacement_packet_found = True
                    else:
                        j += 1
                
                if not replacement_packet_found:
                    assert all(elt is None for elt in self.packets[i:]),\
                        'Packet index renewal failed.'
                    self.packets = self.packets[:i]
                    break

    def merge_packets(self, qubit_vertex: Vertex, packet_index_list: List[int]):
        # Merges all the packets in packet_index_list.
        # This should be followed by a call to self.renew_packet_indices()
        
        packet_index_list.sort()
        first_packet = self.packets[packet_index_list[0]]

        for packet_index in packet_index_list[1:]:
            # Have the first packet absorb the other packets
            intermediate_commands = self.get_intermediate_commands(first_packet.packet_index, packet_index)

            # For reasons I do not understand, the line directly below must be called after the line directly above.
            # Else intermediate_commands becomes a blank list
            first_packet.packet_gate_vertices.extend(self.packets[packet_index].packet_gate_vertices)
            first_packet.intermediate_commands.extend(intermediate_commands)

            # Erase the absorbed packets
            self.packets[packet_index] = None  # Store as None for now to ensure packet referencing doesn't break
            self.packets_by_qubit[first_packet.qubit_vertex].remove(packet_index)
               
    def can_be_merged(self, first_packet_index: int, second_packet_index: int):
        # Check a bunch of conditions to see if two packets can be merged
        # via neighbouring D type packing
        first_packet = self.packets[first_packet_index]
        second_packet = self.packets[second_packet_index]

        if first_packet.qubit_vertex != second_packet.qubit_vertex or first_packet.connected_server_index != second_packet.connected_server_index:
            return False

        intermediate_commands = self.get_intermediate_commands(first_packet_index, second_packet_index)
        for command in intermediate_commands:
            if not is_distributable(command.op):
                return False
        
        return True

    def get_next_packet_index(self, packet_index: int) -> int:
        # Find the next packet index on the same qubit that has the same connected server.
        # Returns None if no such packet exists
        qubit_vertex = self.packets[packet_index].qubit_vertex
        checked_packets = 0
        for potential_packet_index in self.packets_by_qubit[qubit_vertex]:
            if potential_packet_index <= packet_index:
                continue

            if self.packets[potential_packet_index].connected_server_index == self.packets[packet_index].connected_server_index:
                return potential_packet_index

        return None

    def hyperedge_to_packets(self, hyperedge: Hyperedge, starting_index: int) -> Tuple[int, List[Packet]]:
        # Convert a hyperedge into a packet(s)
        # Multiple needed if the hyperedge ends up having gates
        # distributed to multiple servers
        hyperedge_qubit_vertex = self.hypergraph_circuit.get_qubit_vertex(hyperedge)
        connected_server_to_dist_gates: Dict[int, List[Vertex]] = {} #  Server number to list of distributed gates on that server.
        packets: List[Packet] = []
        for gate_vertex in self.hypergraph_circuit.get_gate_vertices(hyperedge):
            connected_server = self.get_connected_server(hyperedge_qubit_vertex, gate_vertex)
            if connected_server in connected_server_to_dist_gates.keys():
                connected_server_to_dist_gates[connected_server].append(gate_vertex)
            else:
                connected_server_to_dist_gates[connected_server] = [gate_vertex]
        
        for connected_server, dist_gates in connected_server_to_dist_gates.items():
            packets.append(
                Packet(starting_index, hyperedge_qubit_vertex, connected_server, dist_gates)
            )
            starting_index += 1
        
        return starting_index, packets

    def get_hyperedges_ordered(self) -> Dict[Vertex, List[Hyperedge]]:
        hyperedges: Dict[Vertex, List[Hyperedge]] = {}

        # Populate the qubit_vertex keys
        for qubit_vertex in self.hypergraph_circuit.get_qubit_vertices():
            hyperedges[qubit_vertex] = []

        # Populate the lists with unsorted hyperedges
        for hyperedge in self.hypergraph_circuit.hyperedge_list:
            qubit_vertex = self.hypergraph_circuit.get_qubit_vertex(hyperedge)
            hyperedges[qubit_vertex].append(hyperedge)

        # Sort the hyperedges (probs room for optimisation here)
        for qubit_vertex in self.hypergraph_circuit.get_qubit_vertices():
            hyperedge_list: List[Hyperedge] = hyperedges[qubit_vertex]
            changes = -1 #  Just a temporary storage value
            while changes != 0:
                changes = 0
                for i in range(len(hyperedge_list) - 1):
                    if (
                        self.get_last_command_index(self.hypergraph_circuit.get_gate_vertices(hyperedge_list[i]))
                        > self.get_first_command_index(self.hypergraph_circuit.get_gate_vertices(hyperedge_list[i+1]))
                    ):
                        hyperedge_list[i], hyperedge_list[i+1] = hyperedge_list[i+1], hyperedge_list[i]
                        changes += 1
            hyperedges[qubit_vertex] = hyperedge_list #  Might be redundant line of code here
        
        return hyperedges

    def get_connected_server(self, qubit_vertex: Vertex, gate_vertex: Vertex):
        command_index = self.vertex_to_command_index[gate_vertex]
        command_dict = self.hypergraph_circuit.commands[command_index]
        qubits = command_dict['command'].qubits
        qubit_vertex_candidates = [self.qubit_vertex_from_qubit(qubit) for qubit in qubits if self.qubit_vertex_from_qubit(qubit) is not qubit_vertex]

        assert len(qubit_vertex_candidates) == 1,\
            'There should be 1 and only 1 other qubit vertex.'
        other_qubit_vertex = qubit_vertex_candidates[0]
        return self.placement.placement[other_qubit_vertex]

    def qubit_vertex_from_qubit(self, qubit: Qubit):
        for vertex, circuit_element\
            in self.hypergraph_circuit.vertex_circuit_map.items():
                if (
                    circuit_element['type'] == 'qubit'
                    and circuit_element['node'] == qubit
                ):
                    return vertex
        raise Exception(
            'Could not find a vertex corresponding to this qubit.'
        )

    def circuit_element_from_vertex(self, vertex: Vertex):
        # This could maybe belong to HypergraphCircuit instead?
        if self.hypergraph_circuit.vertex_circuit_map[vertex]['type'] == 'qubit':
            return self.hypergraph_circuit.vertex_circuit_map[vertex]['node']
        elif self.hypergraph_circuit.vertex_circuit_map[vertex]['type'] == 'gate':
            return self.hypergraph_circuit.vertex_circuit_map[vertex]['command']

    def get_last_command_index(self, gate_vertex_list: List[Vertex]) -> int:
        last_command_index: int = None
        for vertex in gate_vertex_list:
            if last_command_index is None or self.vertex_to_command_index[vertex] > last_command_index:
                last_command_index = self.vertex_to_command_index[vertex]
        
        return last_command_index

    def get_first_command_index(self, gate_vertex_list: List[Vertex]) -> int:
        first_command_index: int = None
        for vertex in gate_vertex_list:
            if first_command_index is None or self.vertex_to_command_index[vertex] < first_command_index:
                first_command_index = self.vertex_to_command_index[vertex]
        
        return first_command_index

    def get_intermediate_commands(self, first_packet_index: int, second_packet_index: int) -> List[Command]:
        first_packet = self.packets[first_packet_index]
        second_packet = self.packets[second_packet_index]

        assert first_packet.qubit_vertex == second_packet.qubit_vertex,\
            'Qubit vertices do not match.'

        qubit_vertex = self.packets[first_packet_index].qubit_vertex
        qubit = self.circuit_element_from_vertex(qubit_vertex)

        last_command_index = self.get_last_command_index(first_packet.packet_gate_vertices)
        first_command_index = self.get_first_command_index(second_packet.packet_gate_vertices)

        intermediate_commands = []

        for command_dict in self.hypergraph_circuit.commands[last_command_index + 1 : first_command_index]:
            command = command_dict['command']
            if qubit in command.qubits:
                intermediate_commands.append(command_dict['command'])

        return intermediate_commands

    def is_embeddable_CU1(self, command: Command, first_packet_index: int, second_packet_index: int) -> bool:
        # Bad function name but this checks that a CU1 itself
        # COULD be embeddable
        # i.e. only prove that we cannot embed but does not check the 1q
        # gates surrounding it, hence doesn't prove it IS embeddable
        qubit_vertex = self.packets[first_packet_index].qubit_vertex
        servers = {
            self.placement.placement[qubit_vertex], self.packets[first_packet_index].connected_server_index
        }

        this_command_qubit_vertices = [self.qubit_vertex_from_qubit(qubit) for qubit in command.qubits]
        this_command_servers = {self.placement.placement[qubit_vertex] for qubit_vertex in this_command_qubit_vertices}
        if servers != this_command_servers:
            return False
        elif not allclose([command.op.params[0]], [1]):
            return False
        return True

    def are_intermediate_commands_embeddable(self, first_packet_index: int, second_packet_index: int) -> bool:
        # Given two packet indices, are the gates between them embeddable?

        # Get the intermediate commands and also a list where they are converted to ops
        intermediate_commands = self.get_intermediate_commands(first_packet_index, second_packet_index)
        intermediate_ops = [
            command.op for command in intermediate_commands
        ]

        distributable_1q_optypes = [
            OpType.Rz,
            OpType.X,
            OpType.U1,
            OpType.Z
        ]

        allowed_optypes = distributable_1q_optypes + [OpType.H, OpType.CU1]

        # Get a list of the indicies in intermediate ops where
        # the op is a CU1
        cu1_op_indicies = [
            i for i, op in enumerate(intermediate_ops)
            if op.type == OpType.CU1
        ]

        for i, cu1_op_index in enumerate(cu1_op_indicies):
            if not self.is_embeddable_CU1(intermediate_commands[cu1_op_index], first_packet_index, second_packet_index):
                # Verify the CU1 can be embedded
                print('Cannot embed CU1')
                return False

            if i == 0:
                prior_1q_ops = intermediate_ops[:cu1_op_index]
                prior_allowed_Hs = 1
            else:
                prior_1q_ops = intermediate_ops[cu1_op_indicies[i-1] + 1: cu1_op_index]
                prior_allowed_Hs = 2
            
            if i == len(cu1_op_indicies) - 1:
                post_1q_ops = intermediate_ops[cu1_op_index + 1:]
                post_allowed_Hs = 1
            else:
                post_1q_ops = intermediate_ops[cu1_op_index + 1: cu1_op_indicies[i + 1]]
                post_allowed_Hs = 2
            
            H_count = 0
            for op in prior_1q_ops:
                if op.type == OpType.H:
                    H_count += 1
                    if H_count > prior_allowed_Hs:
                        print('Too many Hs')
                        return False
                elif op.type not in distributable_1q_optypes:
                    print(f'1q gate {op} not distributable')
                    return False
            
            H_count = 0
            for op in post_1q_ops:
                if op.type == OpType.H:
                    H_count += 1
                    if H_count > post_allowed_Hs:
                        print('Too many Hs')
                        return False
                elif op.type not in distributable_1q_optypes:
                    print(f'1q gate {op} not distributable')
                    return False

            # Convert from 1 Hadamard -> Hadamards as needed for embedding
            # Bit inefficient since checking same conditions as above but needed
            # to verify that the 1q ops are allowed to be distributed
            if i != 0:
                prior_1q_ops = self.convert_1q_ops(prior_1q_ops)
            
            if i != len(cu1_op_indicies) - 1:
                post_1q_ops = self.convert_1q_ops(post_1q_ops)
            
            if not self.are_1q_op_phases_npi(prior_1q_ops, post_1q_ops):
                print('1q phases are not n * pi')
                print(f'Prior 1q gate is {prior_1q_ops[-1]}')
                print(f'Post 1q gate is {post_1q_ops[0]}')
                return False
        
        return True

    def convert_1q_ops(self, ops: List[Op]) -> List[Op]:
        # Converts a set of 1q ops
        # in the gateset of Rz, Z, X, H
        # with up to 1 Hadamard
        # so that it is in the same gateset
        # but has 2 Hadamards in the list.
        # BREAKS IF Z or X or Rz

        hadamard_indices = [
            i for i, op in enumerate(ops)
            if op.type == OpType.H
        ]

        hadamard_count = len(hadamard_indices)

        assert hadamard_count <= 2,\
            'There should not be more than 2 Hadamards.'

        hadamard = Op.create(OpType.H)
        if hadamard_count == 2:
            return ops

        elif hadamard_count == 0:
            ops.insert(0, hadamard)
            ops.append(hadamard)
            return ops
        
        else:
            assert len(ops) <= 3,\
                'There can only be up to 3 ops in this decomposition.'
            new_ops: List[Op] = []
            s_op = Op.create(OpType.U1, [0.5])

            if len(ops) == 1:
                new_ops += [s_op, hadamard, s_op, hadamard, s_op]
            
            elif len(ops == 2):
                phase_op_index = int(not hadamard_indices[0])  # only takes value of 1 or 0
                phase_op = ops[phase_op_index]
                phase = phase_op.params[0]  # phase in turns of pi
                new_phase_op = Op.create(OpType.U1, [phase + 1/2])  # need to add another half phase
                new_ops += [new_phase_op, hadamard, s_op, hadamard, s_op]
                if phase_op_index:
                    new_ops.reverse()

            else:
                first_phase = ops[0].params[0]
                second_phase = ops[2].params[0]
                first_new_phase_op = Op.create(OpType.U1, [first_phase + 0.5])
                second_new_phase_op = Op.create(OpType.U1, [second_phase + 0.5])
                new_ops += [first_new_phase_op, hadamard, s_op, hadamard, second_new_phase_op]
            
            return new_ops

    def are_1q_op_phases_npi(self, prior_1q_ops: List[Op], post_1q_ops: List[Op]) -> bool:
        # Check if the sum of the params of the U1 gates that sandwich a CU1 are
        # equal to n (i.e. the actual phases are equal to n * pi)

        if prior_1q_ops[-1].type == OpType.H:
            prior_phase = 0

        else:
            prior_op = prior_1q_ops[-1]
            prior_phase = prior_op.params[0]

        if post_1q_ops[0].type == OpType.H:
            post_phase = 0

        else:
            post_op = post_1q_ops[0]
            post_phase = post_op.params[0]

        phase_sum = prior_phase + post_phase

        if allclose([phase_sum], [0]):
            return True

        return isclose([floor(phase_sum) / phase_sum], [1])