from pytket_dqc.packing import PacMan, Packet
from pytket_dqc.placement import Placement
from pytket_dqc.circuits import HypergraphCircuit, Hyperedge
from pytket import Circuit, OpType
from pytket.circuit import Op  # type: ignore

cz = Op.create(OpType.CU1, 1)  # For the sake of convinience

# Build a big circuit comprising of the individual
# test circuits stitched together.
subcircuits = []

subplacements = []


def test_build_packets():
    circ = Circuit(3)
    # One Hyperedge but two Packets
    circ.add_gate(cz, [0, 1]).add_gate(cz, [0, 2]).add_gate(cz, [0, 1])
    circ.add_gate(cz, [1, 2]).Rz(1, 1).add_gate(cz, [2, 1])

    placement_dict = {0: 0, 1: 1, 2: 2, 3: 0, 4: 0, 5: 0, 6: 1, 7: 2}
    pacman = PacMan(HypergraphCircuit(circ), Placement(placement_dict))

    H0 = Hyperedge([0, 3, 4, 5])
    H1 = Hyperedge([1, 3, 5, 6])
    H2 = Hyperedge([1, 7])
    H3 = Hyperedge([2, 4, 6, 7])

    P0 = Packet(0, 0, 1, [3, 5], H0)
    P1 = Packet(1, 0, 2, [4], H0)
    P2 = Packet(2, 1, 0, [3, 5], H1)
    P3 = Packet(3, 1, 2, [6], H1)
    P4 = Packet(4, 1, 2, [7], H2)
    P5 = Packet(5, 2, 0, [4], H3)
    P6 = Packet(6, 2, 1, [6, 7], H3)

    packets_by_qubit = {
        0: [P0, P1],
        1: [P2, P3, P4],
        2: [P5, P6]
    }

    assert packets_by_qubit == pacman.packets_by_qubit


def test_identify_neighbouring_packets_00():
    # Custom placement
    placement_dict00 = dict()
    for i in range(6):
        placement_dict00[i] = i // 2
    placement_dict00[6] = 0
    placement_dict00[7] = 0
    placement00 = Placement(placement_dict00)
    subplacements.append(placement00)

    # Build circuit
    circ00 = Circuit(6)
    circ00.add_gate(cz, [0, 2])
    circ00.Rz(0.3, 0)
    circ00.add_gate(cz, [0, 3])
    subcircuits.append(circ00)

    # Make hypergraph of circuit
    h_circ00 = HypergraphCircuit(circ00)

    # Pass into PacMan
    pacman00 = PacMan(h_circ00, placement00)

    # Reference packets
    P0 = pacman00.packets_by_qubit[0][0]
    P1 = pacman00.packets_by_qubit[0][1]
    assert pacman00.neighbouring_packets == {
        0: [(P0, P1)],
        1: [],
        2: [],
        3: [],
        4: [],
        5: [],
    }


def test_identify_neighbouring_packets_01():
    # Custom placement
    placement_dict01 = dict()
    for i in range(6):
        placement_dict01[i] = i // 2
    placement_dict01[6] = 0
    placement_dict01[7] = 0
    placement01 = Placement(placement_dict01)

    # Build circuit
    circ01 = Circuit(6)
    circ01.add_gate(cz, [0, 2])
    circ01.Rz(0.3, 0)
    # Note H Rz(1) H = H Z H = X, an anti-diagonal gate.
    circ01.H(0)
    circ01.Rz(1, 0)
    circ01.H(0)
    circ01.add_gate(cz, [0, 3])
    subcircuits.append(circ01)

    # Make hypergraph of circuit
    h_circ01 = HypergraphCircuit(circ01)

    # Pass into PacMan
    pacman01 = PacMan(h_circ01, placement01)

    # Reference packets
    P0 = pacman01.packets_by_qubit[0][0]
    P1 = pacman01.packets_by_qubit[0][1]
    assert pacman01.neighbouring_packets == {
        0: [(P0, P1)],
        1: [],
        2: [],
        3: [],
        4: [],
        5: [],
    }


def test_identify_neighbouring_packets_02():
    # Custom placement
    placement_dict02 = dict()
    for i in range(6):
        placement_dict02[i] = i // 2
    placement_dict02[6] = 0
    placement_dict02[7] = 0
    placement_dict02[8] = 0
    placement_dict02[9] = 0
    placement02 = Placement(placement_dict02)

    # Build circuit
    circ02 = Circuit(6)
    circ02.add_gate(cz, [0, 2])
    circ02.add_gate(cz, [0, 4])
    circ02.Rz(0.5, 0)
    circ02.add_gate(cz, [0, 3])
    circ02.add_gate(cz, [0, 5])
    subcircuits.append(circ02)

    # Make hypergraph of circuit
    h_circ02 = HypergraphCircuit(circ02)

    # Pass into PacMan
    pacman02 = PacMan(h_circ02, placement02)

    P0 = pacman02.packets_by_qubit[0][0]
    P1 = pacman02.packets_by_qubit[0][1]
    P2 = pacman02.packets_by_qubit[0][2]
    P3 = pacman02.packets_by_qubit[0][3]

    assert pacman02.neighbouring_packets == {
        0: [(P0, P2), (P1, P3)],
        1: [],
        2: [],
        3: [],
        4: [],
        5: [],
    }


def test_identify_hopping_packets_03():
    # Custom placement
    placement_dict03 = dict()
    for i in range(6):
        placement_dict03[i] = i // 2
    placement_dict03[6] = 0
    placement_dict03[7] = 0
    placement_dict03[8] = 0
    placement03 = Placement(placement_dict03)
    subplacements.append(placement03)

    # Build circuit
    circ03 = Circuit(6)
    circ03.add_gate(cz, [0, 2])
    circ03.H(0)
    circ03.add_gate(cz, [0, 3])
    circ03.H(0)
    circ03.add_gate(cz, [0, 2])
    subcircuits.append(circ03)

    # Make hypergraph of circuit
    h_circ03 = HypergraphCircuit(circ03)

    # Pass into PacMan
    pacman03 = PacMan(h_circ03, placement03)

    # Reference Packets
    P0 = pacman03.packets_by_qubit[0][0]
    P2 = pacman03.packets_by_qubit[0][2]

    assert pacman03.hopping_packets == {
        0: [(P0, P2)],
        1: [],
        2: [],
        3: [],
        4: [],
        5: [],
    }


def test_identify_hopping_packets_04():
    # Custom placement
    placement_dict04 = dict()
    for i in range(6):
        placement_dict04[i] = i // 2
    placement_dict04[6] = 0
    placement_dict04[7] = 0
    placement_dict04[8] = 0
    placement_dict04[9] = 0
    placement04 = Placement(placement_dict04)
    subplacements.append(placement04)

    # Build circuit
    circ04 = Circuit(6)
    circ04.add_gate(cz, [0, 2])
    circ04.H(0)
    circ04.add_gate(cz, [0, 2])
    circ04.Rz(1, 0)
    circ04.add_gate(cz, [0, 3])
    circ04.H(0)
    circ04.add_gate(cz, [0, 2])
    subcircuits.append(circ04)

    # Make hypergraph of circuit
    h_circ04 = HypergraphCircuit(circ04)

    # Pass into PacMan
    pacman04 = PacMan(h_circ04, placement04)

    P0 = pacman04.packets_by_qubit[0][0]
    P3 = pacman04.packets_by_qubit[0][3]

    assert pacman04.hopping_packets == {
        0: [(P0, P3)],
        1: [],
        2: [],
        3: [],
        4: [],
        5: [],
    }


def test_identify_hopping_packets_05():
    # Custom placement
    placement_dict05 = dict()
    for i in range(6):
        placement_dict05[i] = i // 2
    placement_dict05[6] = 0
    placement_dict05[7] = 0
    placement_dict05[8] = 0
    placement_dict05[9] = 0
    placement05 = Placement(placement_dict05)
    subplacements.append(placement05)

    # Build circuit
    circ05 = Circuit(6)
    circ05.add_gate(cz, [0, 2])
    circ05.H(0)
    circ05.add_gate(cz, [0, 2])
    # Note H Rz(1) H = H Z H = X, an anti-diagonal gate.
    circ05.H(0)
    circ05.Rz(1, 0)
    circ05.H(0)
    circ05.add_gate(cz, [0, 3])
    circ05.H(0)
    circ05.add_gate(cz, [0, 2])
    subcircuits.append(circ05)

    # Make hypergraph of circuit
    h_circ05 = HypergraphCircuit(circ05)

    # Pass into PacMan
    pacman05 = PacMan(h_circ05, placement05)

    P0 = pacman05.packets_by_qubit[0][0]
    P3 = pacman05.packets_by_qubit[0][3]

    assert pacman05.hopping_packets == {
        0: [(P0, P3)],
        1: [],
        2: [],
        3: [],
        4: [],
        5: [],
    }


def test_identify_hopping_packets_06():
    # Custom placement
    placement_dict06 = dict()
    for i in range(6):
        placement_dict06[i] = i // 2
    placement_dict06[6] = 0
    placement_dict06[7] = 0
    placement_dict06[8] = 0
    placement_dict06[9] = 0
    placement06 = Placement(placement_dict06)
    subplacements.append(placement06)

    # Build circuit
    circ06 = Circuit(6)
    circ06.add_gate(cz, [0, 2])
    circ06.H(0)
    circ06.add_gate(cz, [0, 2])
    circ06.H(0)
    circ06.Rz(0.27, 0)
    circ06.H(0)
    circ06.add_gate(cz, [0, 3])
    circ06.H(0)
    circ06.add_gate(cz, [0, 2])
    subcircuits.append(circ06)

    # Make hypergraph of circuit
    h_circ06 = HypergraphCircuit(circ06)

    # Pass into PacMan
    pacman06 = PacMan(h_circ06, placement06)

    P0 = pacman06.packets_by_qubit[0][0]
    P3 = pacman06.packets_by_qubit[0][3]

    assert pacman06.hopping_packets == {
        0: [(P0, P3)],
        1: [],
        2: [],
        3: [],
        4: [],
        5: [],
    }


def test_identify_hopping_packets_07():
    # Custom placement
    placement_dict07 = dict()
    for i in range(6):
        placement_dict07[i] = i // 2
    placement_dict07[6] = 0
    placement_dict07[7] = 0
    placement_dict07[8] = 0
    placement_dict07[9] = 0
    placement07 = Placement(placement_dict07)
    subplacements.append(placement07)

    # Build circuit
    circ07 = Circuit(6)
    circ07.add_gate(cz, [0, 2])
    circ07.H(0)
    circ07.add_gate(cz, [0, 2])
    circ07.Rz(0.5, 0)
    circ07.H(0)
    circ07.Rz(0.5, 0)
    circ07.add_gate(cz, [0, 3])
    circ07.H(0)
    circ07.add_gate(cz, [0, 2])
    subcircuits.append(circ07)

    # Make hypergraph of circuit
    h_circ07 = HypergraphCircuit(circ07)

    # Pass into PacMan
    pacman07 = PacMan(h_circ07, placement07)

    P0 = pacman07.packets_by_qubit[0][0]
    P3 = pacman07.packets_by_qubit[0][3]

    assert pacman07.hopping_packets == {
        0: [(P0, P3)],
        1: [],
        2: [],
        3: [],
        4: [],
        5: [],
    }


def test_identify_hopping_packets_08():
    # Custom placement
    placement_dict08 = dict()
    for i in range(6):
        placement_dict08[i] = i // 2
    placement_dict08[6] = 0
    placement_dict08[7] = 0
    placement_dict08[8] = 0
    placement_dict08[9] = 0
    placement08 = Placement(placement_dict08)
    subplacements.append(placement08)

    # Build circuit
    circ08 = Circuit(6)
    circ08.add_gate(cz, [0, 2])
    circ08.H(0)
    circ08.Rz(0.33, 0)
    circ08.add_gate(cz, [0, 2])
    circ08.Rz(0.67, 0)
    circ08.H(0)
    circ08.Rz(0.27, 0)
    circ08.H(0)
    circ08.Rz(0.43847, 0)
    circ08.add_gate(cz, [0, 3])
    circ08.Rz(1 - 0.43847, 0)
    circ08.H(0)
    circ08.add_gate(cz, [0, 2])
    subcircuits.append(circ08)

    # Make hypergraph of circuit
    h_circ08 = HypergraphCircuit(circ08)

    # Pass into PacMan
    pacman08 = PacMan(h_circ08, placement08)

    P0 = pacman08.packets_by_qubit[0][0]
    P3 = pacman08.packets_by_qubit[0][3]

    assert pacman08.hopping_packets == {
        0: [(P0, P3)],
        1: [],
        2: [],
        3: [],
        4: [],
        5: [],
    }


def test_identify_hopping_packets_09():
    # Custom placement
    placement_dict09 = dict()
    for i in range(6):
        placement_dict09[i] = i // 2
    placement_dict09[6] = 0
    placement_dict09[7] = 0
    placement_dict09[8] = 0
    placement_dict09[9] = 0
    placement09 = Placement(placement_dict09)
    subplacements.append(placement09)

    # Build circuit
    circ09 = Circuit(6)
    circ09.add_gate(cz, [0, 2])
    circ09.H(0)
    circ09.add_gate(cz, [0, 2])
    circ09.Rz(1, 0)
    circ09.add_gate(cz, [0, 4])
    circ09.H(0)
    circ09.add_gate(cz, [0, 2])
    subcircuits.append(circ09)

    # Make hypergraph of circuit
    h_circ09 = HypergraphCircuit(circ09)

    # Pass into PacMan
    pacman09 = PacMan(h_circ09, placement09)

    assert pacman09.hopping_packets == {
        0: [],
        1: [],
        2: [],
        3: [],
        4: [],
        5: [],
    }


def test_identify_hopping_packets_10():
    # Custom placement
    placement_dict10 = dict()
    for i in range(6):
        placement_dict10[i] = i // 2
    placement_dict10[6] = 0
    placement_dict10[7] = 0
    placement_dict10[8] = 0
    placement_dict10[9] = 0
    placement10 = Placement(placement_dict10)
    subplacements.append(placement10)

    # Build circuit
    circ10 = Circuit(6)
    circ10.add_gate(cz, [0, 2])
    circ10.H(0)
    circ10.add_gate(cz, [0, 2])
    circ10.Rz(1, 0)
    circ10.add_gate(cz, [0, 1])
    circ10.H(0)
    circ10.add_gate(cz, [0, 2])
    subcircuits.append(circ10)

    # Make hypergraph of circuit
    h_circ10 = HypergraphCircuit(circ10)

    # Pass into PacMan
    pacman10 = PacMan(h_circ10, placement10)

    assert pacman10.hopping_packets == {
        0: [],
        1: [],
        2: [],
        3: [],
        4: [],
        5: [],
    }


def test_merge_packets():

    big_circ = Circuit(6)
    for subcircuit in subcircuits:
        big_circ.add_circuit(subcircuit, list(range(6)))
    big_h_circ = HypergraphCircuit(big_circ)

    big_placement = {}
    for i in range(6):
        big_placement[i] = i // 2

    current_key = 6
    for subplacement in subplacements:
        for i in sorted([j for j in subplacement.placement.keys() if j > 5]):
            big_placement[current_key] = subplacement.placement[i]
            current_key += 1
    pacman = PacMan(big_h_circ, Placement(big_placement))

    # Reference Packets
    P0 = pacman.packets_by_qubit[0][0]
    P1 = pacman.packets_by_qubit[0][1]
    P2 = pacman.packets_by_qubit[0][2]
    P3 = pacman.packets_by_qubit[0][3]
    P4 = pacman.packets_by_qubit[0][4]
    P5 = pacman.packets_by_qubit[0][5]
    P6 = pacman.packets_by_qubit[0][6]
    P7 = pacman.packets_by_qubit[0][7]
    P8 = pacman.packets_by_qubit[0][8]
    P9 = pacman.packets_by_qubit[0][9]
    P10 = pacman.packets_by_qubit[0][10]
    P11 = pacman.packets_by_qubit[0][11]
    P12 = pacman.packets_by_qubit[0][12]
    P13 = pacman.packets_by_qubit[0][13]
    P14 = pacman.packets_by_qubit[0][14]
    P15 = pacman.packets_by_qubit[0][15]
    P16 = pacman.packets_by_qubit[0][16]
    P17 = pacman.packets_by_qubit[0][17]
    P18 = pacman.packets_by_qubit[0][18]
    P19 = pacman.packets_by_qubit[0][19]
    P20 = pacman.packets_by_qubit[0][20]
    P21 = pacman.packets_by_qubit[0][21]
    P22 = pacman.packets_by_qubit[0][22]
    P23 = pacman.packets_by_qubit[0][23]
    P24 = pacman.packets_by_qubit[0][24]
    P25 = pacman.packets_by_qubit[0][25]
    P26 = pacman.packets_by_qubit[0][26]
    P27 = pacman.packets_by_qubit[0][27]

    P28 = pacman.packets_by_qubit[2][0]
    P29 = pacman.packets_by_qubit[3][0]
    P30 = pacman.packets_by_qubit[4][0]
    P31 = pacman.packets_by_qubit[5][0]

    merged_packets_ref = {
        0: [
            (P0, P1, P2, P4, P7, P10, P13, P16, P19, P22),
            (P3, P5),
            (P6, P8, P9, P11, P12, P14),
            (P15, P17),
            (P18, P20),
            (P21, P23, P26),
            (P24,),
            (P25,),
            (P27,),
        ],
        1: [],
        2: [(P28,)],
        3: [(P29,)],
        4: [(P30,)],
        5: [(P31,)],
    }
    assert merged_packets_ref == pacman.merged_packets


def test_intertwining_embeddings_0():
    """Test that two sequential hopping packets
    are identified correctly.
    """
    circ = Circuit(5)
    circ.add_gate(cz, [0, 1]).H(0).add_gate(cz, [0, 2]).H(0)
    circ.add_gate(cz, [0, 3]).H(0).add_gate(cz, [0, 4])
    placement_dict = {0: 0, 1: 1, 2: 1, 3: 1, 4: 1, 5: 0, 6: 0, 7: 0, 8: 0}
    placement = Placement(placement_dict)

    hyp_circ = HypergraphCircuit(circ)
    pacman = PacMan(hyp_circ, placement)

    P0 = pacman.packets_by_qubit[0][0]
    P1 = pacman.packets_by_qubit[0][1]
    P2 = pacman.packets_by_qubit[0][2]
    P3 = pacman.packets_by_qubit[0][3]
    P4 = pacman.packets_by_qubit[1][0]
    P5 = pacman.packets_by_qubit[2][0]
    P6 = pacman.packets_by_qubit[3][0]
    P7 = pacman.packets_by_qubit[4][0]

    merged_packets_ref = {
        0: [(P0, P2), (P1, P3)],
        1: [(P4,)],
        2: [(P5,)],
        3: [(P6,)],
        4: [(P7,)],
    }
    assert merged_packets_ref == pacman.merged_packets


def test_intertwining_embeddings_1():
    circ = Circuit(6)
    circ.add_gate(cz, [0, 1]).H(0).add_gate(cz, [0, 2]).H(0)
    circ.add_gate(cz, [0, 3]).H(0).add_gate(cz, [0, 4]).H(0).add_gate(
        cz, [0, 5]
    )
    placement_dict = {
        0: 0,
        1: 1,
        2: 1,
        3: 1,
        4: 1,
        5: 1,
        6: 0,
        7: 0,
        8: 0,
        9: 0,
        10: 0,
    }
    placement = Placement(placement_dict)

    hyp_circ = HypergraphCircuit(circ)
    pacman = PacMan(hyp_circ, placement)

    P0 = pacman.packets_by_qubit[0][0]
    P1 = pacman.packets_by_qubit[0][1]
    P2 = pacman.packets_by_qubit[0][2]
    P3 = pacman.packets_by_qubit[0][3]
    P4 = pacman.packets_by_qubit[0][4]
    P5 = pacman.packets_by_qubit[1][0]
    P6 = pacman.packets_by_qubit[2][0]
    P7 = pacman.packets_by_qubit[3][0]
    P8 = pacman.packets_by_qubit[4][0]
    P9 = pacman.packets_by_qubit[5][0]

    merged_packets_ref = {
        0: [(P0, P2, P4), (P1, P3)],
        1: [(P5,)],
        2: [(P6,)],
        3: [(P7,)],
        4: [(P8,)],
        5: [(P9,)],
    }
    assert merged_packets_ref == pacman.merged_packets
