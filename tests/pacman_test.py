from pytket_dqc.packing import PacMan, Packet
from pytket_dqc.placement import Placement
from pytket_dqc.circuits import HypergraphCircuit, Vertex
from pytket import Circuit, OpType
from pytket.circuit import Op

test_circuit = Circuit(6)
# This test circuit is comprised of sections
# designed to test various things

# Test that hyperedges on different servers are split
# Test that hyperedges split by (anti)diagonal gates
# are merged
cz = Op.create(OpType.CU1, 1)
test_circuit.add_gate(cz, [0,2])
test_circuit.add_gate(cz, [0,4])
test_circuit.Z(0).X(0)
test_circuit.add_gate(cz, [0,5])
test_circuit.add_gate(cz, [0,3])

# Test we can embed two CU1s with no Hadamard
# one Hadamard and two Hadamards
# S gates inserted to ensure angle of phase gates
# sum to integer
test_circuit.H(0)
test_circuit.Rz(0.5, 0)
test_circuit.add_gate(cz, [0,2])
test_circuit.Rz(0.5, 0) # 0 H
test_circuit.add_gate(cz, [0,3])
test_circuit.H(0) # 1 H
test_circuit.add_gate(cz, [0,2]) # NOT mergeable
test_circuit.Rz(0.5, 0)
test_circuit.H(0) # 2 H
test_circuit.Rz(0.27, 0) # Random phase that should have no effect
test_circuit.H(0)
test_circuit.add_gate(cz, [0,2])
test_circuit.H(0)
test_circuit.add_gate(cz, [0,3]) # This gate is mergeable

# Test that local and 3rd party server CU1s break embeddability
test_circuit.H(0)
test_circuit.add_gate(cz, [0,1]) # Local CU1
test_circuit.H(0)
test_circuit.add_gate(cz, [0,2]) # NOT mergeable
test_circuit.H(0)
test_circuit.add_gate(cz, [0,4]) # 3rd party CU1
test_circuit.H(0)
test_circuit.add_gate(cz, [0,2]) # NOT mergeable

placement_dict = {
    0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2,
}
for i in range(6, 19):
    placement_dict[i] = 0

placement = Placement(placement_dict)

hypergraph_circuit = HypergraphCircuit(test_circuit)

def test_build_vertex_to_command_index():
    pacman = PacMan(hypergraph_circuit, placement)
    command_indices = [
        0,1,4,5,8,10,12,17,19,21,23,25,27
    ]
    vertex_to_command_index_reference = dict()
    for i in range(6, 19):
        vertex_to_command_index_reference[i] = command_indices[i - 6]
    assert pacman.vertex_to_command_index == vertex_to_command_index_reference

def test_build_packets():
    packets_by_qubit_reference = dict()
    q0_packets = [
        Packet(0, 0, 1, [6]),
        Packet(1, 0, 2, [7]),
        Packet(2, 0, 2, [8]),
        Packet(3, 0, 1, [9]),
        Packet(4, 0, 1, [10]),
        Packet(5, 0, 1, [11]),
        Packet(6, 0, 1, [12]),
        Packet(7, 0, 1, [13]),
        Packet(8, 0, 1, [14]),
        Packet(9, 0, 0, [15]),
        Packet(10, 0, 1, [16]),
        Packet(11, 0, 2, [17]),
        Packet(12, 0, 1, [18]),
    ]

    q1_packets = [
        Packet(13, 1, 0, [15])
    ]

    q2_packets = [
        Packet(14, 2, 0, [6,10,12,13,16,18])
    ]

    q3_packets = [
        Packet(15, 3, 0, [9,11,14])
    ]

    q4_packets = [
        Packet(16, 4, 0, [7,17])
    ]

    q5_packets = [
        Packet(17, 5, 0, [8])
    ]

    packets_by_qubit_reference[0] = q0_packets
    packets_by_qubit_reference[1] = q1_packets
    packets_by_qubit_reference[2] = q2_packets
    packets_by_qubit_reference[3] = q3_packets
    packets_by_qubit_reference[4] = q4_packets
    packets_by_qubit_reference[5] = q5_packets
    
    pacman = PacMan(hypergraph_circuit, placement)
    pacman.packets_by_qubit = dict()  # Erase the packets since they'll be merged
    pacman.build_packets()  # Rebuild prior to merging
    assert pacman.packets_by_qubit == packets_by_qubit_reference

def test_identify_and_merge_neighbouring_packets():
    packets_by_qubit_reference = dict()
    q0_packets = [
        Packet(0, 0, 1, [6,9]),
        Packet(1, 0, 2, [7,8]),
        Packet(2, 0, 1, [10,11]),
        Packet(3, 0, 1, [12]),
        Packet(4, 0, 1, [13]),
        Packet(5, 0, 1, [14]),
        Packet(6, 0, 0, [15]),
        Packet(7, 0, 1, [16]),
        Packet(8, 0, 2, [17]),
        Packet(9, 0, 1, [18]),
    ]

    q1_packets = [
        Packet(10, 1, 0, [15])
    ]

    q2_packets = [
        Packet(11, 2, 0, [6,10,12,13,16,18])
    ]

    q3_packets = [
        Packet(12, 3, 0, [9,11,14])
    ]

    q4_packets = [
        Packet(13, 4, 0, [7,17])
    ]

    q5_packets = [
        Packet(14, 5, 0, [8])
    ]

    packets_by_qubit_reference[0] = q0_packets
    packets_by_qubit_reference[1] = q1_packets
    packets_by_qubit_reference[2] = q2_packets
    packets_by_qubit_reference[3] = q3_packets
    packets_by_qubit_reference[4] = q4_packets
    packets_by_qubit_reference[5] = q5_packets

    pacman = PacMan(hypergraph_circuit, placement)
    for key, value in packets_by_qubit_reference.items():
        assert value == pacman.packets_by_qubit[key]
