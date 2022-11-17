from pytket_dqc.packing import PacMan, Packet
from pytket_dqc.placement import Placement
from pytket_dqc.circuits import HypergraphCircuit
from pytket import Circuit, OpType
from pytket.circuit import Op  # type: ignore

cz = Op.create(OpType.CU1, 1)  # For the sake of convinience

# Below are a bunch of a circuits with certain
# desirable qualities from a testing POV.
# Placements and HypergraphCircuits for each part
# are also given.
# They are in the global scope because they are
# used in a test that has _XX appended at the
# end ``test_build_packets``, and ``test_merge_packets``.
# Each part can be viewed visually in the ``pacman-example.ipynb``.

# Build the big circuit comprising of these little parts.
subcircuits = []

subplacements = []


def test_build_packets():
    assert True


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
    p0 = Packet(0, 0, 1, [6])
    p1 = Packet(1, 0, 1, [7])
    assert pacman00.neighbouring_packets == {
        0: [(p0, p1)],
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
    p0 = Packet(0, 0, 1, [6])
    p1 = Packet(1, 0, 1, [7])
    assert pacman01.neighbouring_packets == {
        0: [(p0, p1)],
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

    p0 = Packet(0, 0, 1, [6])
    p1 = Packet(1, 0, 2, [7])
    p2 = Packet(2, 0, 1, [8])
    p3 = Packet(3, 0, 2, [9])

    assert pacman02.neighbouring_packets == {
        0: [(p0, p2), (p1, p3)],
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
    p0 = Packet(0, 0, 1, [6])
    # p1 = Packet(1, 0, 1, [7])
    p2 = Packet(2, 0, 1, [8])

    assert pacman03.hopping_packets == {
        0: [(p0, p2)],
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

    p0 = Packet(0, 0, 1, [6])
    p3 = Packet(3, 0, 1, [9])

    assert pacman04.hopping_packets == {
        0: [(p0, p3)],
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

    p0 = Packet(0, 0, 1, [6])
    p3 = Packet(3, 0, 1, [9])

    assert pacman05.hopping_packets == {
        0: [(p0, p3)],
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

    p0 = Packet(0, 0, 1, [6])
    p3 = Packet(3, 0, 1, [9])

    assert pacman06.hopping_packets == {
        0: [(p0, p3)],
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

    p0 = Packet(0, 0, 1, [6])
    p3 = Packet(3, 0, 1, [9])

    assert pacman07.hopping_packets == {
        0: [(p0, p3)],
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

    p0 = Packet(0, 0, 1, [6])
    p3 = Packet(3, 0, 1, [9])

    assert pacman08.hopping_packets == {
        0: [(p0, p3)],
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
    P0 = Packet(0, 0, 1, [6])
    P1 = Packet(1, 0, 1, [7, 8])
    P2 = Packet(2, 0, 1, [9, 10])
    P3 = Packet(3, 0, 2, [11])
    P4 = Packet(4, 0, 1, [12, 14])
    P5 = Packet(5, 0, 2, [13])
    P6 = Packet(6, 0, 1, [15])
    P7 = Packet(7, 0, 1, [16, 17])
    P8 = Packet(8, 0, 1, [18])
    P9 = Packet(9, 0, 1, [19])
    P10 = Packet(10, 0, 1, [20, 21])
    P11 = Packet(11, 0, 1, [22])
    P12 = Packet(12, 0, 1, [23])
    P13 = Packet(13, 0, 1, [24, 25])
    P14 = Packet(14, 0, 1, [26])
    P15 = Packet(15, 0, 1, [27])
    P16 = Packet(16, 0, 1, [28, 29])
    P17 = Packet(17, 0, 1, [30])
    P18 = Packet(18, 0, 1, [31])
    P19 = Packet(19, 0, 1, [32, 33])
    P20 = Packet(20, 0, 1, [34])
    P21 = Packet(21, 0, 1, [35])
    P22 = Packet(22, 0, 1, [36, 37])
    P23 = Packet(23, 0, 1, [38])
    P24 = Packet(24, 0, 2, [39])
    P25 = Packet(25, 0, 1, [40, 41])
    P26 = Packet(26, 0, 1, [42])
    P27 = Packet(27, 0, 1, [44])
    P28 = Packet(
        28,
        2,
        0,
        [
            6,
            8,
            10,
            14,
            16,
            17,
            18,
            20,
            21,
            22,
            24,
            25,
            26,
            28,
            29,
            30,
            32,
            33,
            34,
            36,
            37,
            38,
            40,
            41,
            42,
            44,
        ],
    )
    P29 = Packet(
        29,
        3,
        0,
        [
            7,
            9,
            12,
            15,
            19,
            23,
            27,
            31,
            35,
        ],
    )
    P30 = Packet(30, 4, 0, [11, 39])
    P31 = Packet(31, 5, 0, [13])

    merged_packets_ref = {
        0: [
            (P0, P1, P2, P4, P7, P10, P13, P16, P19, P22),
            (P3, P5),
            (P6, P8, P9, P11, P12, P14),
            (P15, P17),
            (P18, P20),
            (P21, P23),
            (P24,),
            (P25,),
            (P26,),
            (P27,),
        ],
        1: [],
        2: [(P28,)],
        3: [(P29,)],
        4: [(P30,)],
        5: [(P31,)],
    }

    assert merged_packets_ref == pacman.merged_packets
