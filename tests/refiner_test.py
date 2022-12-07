import pytest
import json  # type: ignore
from pytket import Circuit, OpType  # type: ignore
from pytket_dqc import Distribution
from pytket_dqc.circuits import HypergraphCircuit
from pytket_dqc.circuits.hypergraph import Hyperedge
from pytket_dqc.placement import Placement
from pytket_dqc.utils import check_equivalence, DQCPass
from pytket_dqc.networks import NISQNetwork
from pytket_dqc.refiners.vertex_cover import (
    VertexCover,
    get_min_covers,
)
from pytket_dqc.refiners import SequentialDTypeMerge


def test_min_covers():
    edges = [(0,1), (0,2), (0,5), (1,3), (1,4), (2,5), (3,4), (4,5)]
    covers = [
        {0,1,2,4},
        {0,1,5,3},
        {0,1,5,4},
        {0,3,4,2},
        {0,3,4,5},
        {1,2,5,3},
        {1,2,5,4},
    ]
    assert sorted(covers) == sorted(get_min_covers(edges))

    edges = [(0,1), (0,2), (0,5), (1,3), (1,4), (2,4), (3,4), (4,5)]
    covers = [
        {0,1,4},
        {0,3,4},
    ]
    assert sorted(covers) == sorted(get_min_covers(edges))


def test_vertex_cover_refiner_empty():

    network = NISQNetwork([[0,1]], {0: [0, 1], 1: [2]})

    circ = Circuit(2)

    hyp_circ = HypergraphCircuit(circ)
    placement = Placement({0: 0, 1: 0, 2: 0})

    distribution = Distribution(
        circuit=hyp_circ, placement=placement, network=network)

    VertexCover().refine(distribution, vertex_cover_alg="all_brute_force")

    pytket_circ = distribution.to_pytket_circuit()

    assert check_equivalence(
        circ, pytket_circ, distribution.get_qubit_mapping()
    )


def test_vertex_cover_refiner_trivial():

    network = NISQNetwork([[0,1]], {0: [0, 1], 1: [2]})

    circ = Circuit(2)
    circ.add_gate(OpType.CU1, 1.0, [0, 1])

    hyp_circ = HypergraphCircuit(circ)
    placement = Placement({0: 0, 1: 0, 2: 0})

    distribution = Distribution(
        circuit=hyp_circ, placement=placement, network=network)

    VertexCover().refine(distribution, vertex_cover_alg="all_brute_force")

    pytket_circ = distribution.to_pytket_circuit()

    assert check_equivalence(
        circ, pytket_circ, distribution.get_qubit_mapping()
    )


def test_vertex_cover_refiner_simple():
    network = NISQNetwork(
        [[2, 1], [1, 0], [1, 3], [0, 4]],
        {0: [0], 1: [1], 2: [2], 3: [3], 4: [4, 5]},
    )

    circ = (
        Circuit(3)
        .add_gate(OpType.CU1, 0.3, [0, 1])
        .H(0)
        .Rz(1.0, 0)
        .H(0)
        .add_gate(OpType.CU1, 0.8, [0, 2])
    )
    hyp_circ = HypergraphCircuit(circ)
    placement = Placement({0: 0, 1: 4, 2: 4})
    distribution = Distribution(
        circuit=hyp_circ, placement=placement, network=network)

    VertexCover().refine(distribution, vertex_cover_alg="all_brute_force")

    pytket_circ = distribution.to_pytket_circuit()

    assert check_equivalence(
        circ, pytket_circ, distribution.get_qubit_mapping()
    )


def test_vertex_cover_refiner_complex_1():

    network = NISQNetwork(
        server_coupling=[[0, 1], [1, 2], [1, 3]],
        server_qubits={0: [0], 1: [1], 2: [2], 3: [3]}
    )

    circ = Circuit(4)
    circ.add_gate(OpType.CU1, 1.0, [0, 2])
    circ.Rz(0.3, 0)
    circ.add_gate(OpType.CU1, 1.0, [0, 1])
    circ.H(2)
    circ.add_gate(OpType.CU1, 1.0, [3, 2])
    circ.H(2)
    circ.H(0)
    circ.add_gate(OpType.CU1, 1.0, [0, 2])
    circ.add_gate(OpType.CU1, 1.0, [3, 0])
    circ.H(0)
    circ.add_gate(OpType.CU1, 1.0, [1, 0])

    hyp_circ = HypergraphCircuit(circ)
    placement = Placement({0: 0, 1: 1, 2: 2, 3: 3})
    distribution = Distribution(
        circuit=hyp_circ, placement=placement, network=network)

    VertexCover().refine(distribution, vertex_cover_alg="all_brute_force")

    pytket_circ = distribution.to_pytket_circuit()

    assert check_equivalence(
        circ, pytket_circ, distribution.get_qubit_mapping()
    )


def test_vertex_cover_refiner_complex_2():
    network = NISQNetwork(
        [[0, 1], [0, 2], [0, 3], [3, 4]],
        {0: [0], 1: [1, 2], 2: [3, 4], 3: [7], 4: [5, 6]},
    )

    circ = Circuit(4)
    circ.add_gate(OpType.CU1, 0.1234, [1, 2])
    circ.add_gate(OpType.CU1, 0.1234, [0, 2])
    circ.add_gate(OpType.CU1, 0.1234, [2, 3])
    circ.add_gate(OpType.CU1, 0.1234, [0, 3])
    circ.H(0).H(2).Rz(0.1234, 3)
    circ.add_gate(OpType.CU1, 1.0, [0, 2])
    circ.add_gate(OpType.CU1, 1.0, [0, 3])
    circ.add_gate(OpType.CU1, 1.0, [1, 2])
    circ.H(0).H(2).Rz(0.1234, 0)
    circ.add_gate(OpType.CU1, 0.1234, [0, 1])
    circ.add_gate(OpType.CU1, 0.1234, [0, 3])
    circ.add_gate(OpType.CU1, 1.0, [1, 2])

    hyp_circ = HypergraphCircuit(circ)
    placement = Placement({0: 1, 1: 1, 2: 2, 3: 4})
    distribution = Distribution(
        circuit=hyp_circ, placement=placement, network=network)

    VertexCover().refine(distribution, vertex_cover_alg="all_brute_force")

    pytket_circ = distribution.to_pytket_circuit()

    assert check_equivalence(
        circ, pytket_circ, distribution.get_qubit_mapping()
    )


def test_vertex_cover_refiner_pauli_circ():
    # Randomly generated circuit of type pauli, depth 10 and 10 qubits
    with open(
        "tests/test_circuits/to_pytket_circuit/pauli_10.json", "r"
    ) as fp:
        circ = Circuit().from_dict(json.load(fp))

    DQCPass().apply(circ)

    network = NISQNetwork(
        [[2, 1], [1, 0], [1, 3], [0, 4]],
        {0: [0, 1, 2], 1: [3, 4], 2: [5, 6, 7], 3: [8], 4: [9]},
    )

    hyp_circ = HypergraphCircuit(circ)
    placement = Placement({0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 2, 6: 2, 7: 2, 8: 3, 9: 4})
    distribution = Distribution(
        circuit=hyp_circ, placement=placement, network=network)

    VertexCover().refine(distribution, vertex_cover_alg="all_brute_force")

    pytket_circ = distribution.to_pytket_circuit()

    assert check_equivalence(
        circ, pytket_circ, distribution.get_qubit_mapping()
    )


def test_vertex_cover_refiner_random_circ():
    # Randomly generated circuit of type random, depth 6 and 6 qubits
    with open(
        "tests/test_circuits/to_pytket_circuit/random_6.json", "r"
    ) as fp:
        circ = Circuit().from_dict(json.load(fp))

    DQCPass().apply(circ)

    network = NISQNetwork(
        [[2, 1], [1, 0], [1, 3], [0, 4]],
        {0: [0, 1, 2], 1: [3, 4], 2: [5, 6, 7], 3: [8], 4: [9]},
    )

    hyp_circ = HypergraphCircuit(circ)
    placement = Placement({0: 0, 1: 4, 2: 3, 3: 2, 4: 1, 5: 2})
    distribution = Distribution(
        circuit=hyp_circ, placement=placement, network=network)

    VertexCover().refine(distribution, vertex_cover_alg="all_brute_force")

    pytket_circ = distribution.to_pytket_circuit()

    assert check_equivalence(
        circ, pytket_circ, distribution.get_qubit_mapping()
    )


def test_vertex_cover_refiner_frac_CZ_circ():
    # Randomly generated circuit of type frac_CZ, depth 10 and 10 qubits
    with open(
        "tests/test_circuits/to_pytket_circuit/frac_CZ_10.json", "r"
    ) as fp:
        circ = Circuit().from_dict(json.load(fp))

    DQCPass().apply(circ)

    network = NISQNetwork(
        [[2, 1], [1, 0], [1, 3], [0, 4]],
        {0: [0, 1, 2], 1: [3, 4], 2: [5, 6, 7], 3: [8], 4: [9]},
    )

    hyp_circ = HypergraphCircuit(circ)
    placement = Placement({0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 2, 6: 2, 7: 2, 8: 3, 9: 4})
    distribution = Distribution(
        circuit=hyp_circ, placement=placement, network=network)

    VertexCover().refine(distribution, vertex_cover_alg="all_brute_force")

    pytket_circ = distribution.to_pytket_circuit()

    assert check_equivalence(
        circ, pytket_circ, distribution.get_qubit_mapping()
    )


def test_sequential_merge_d_type_backwards_meregable():
    # Note that this test identifies the limits of SequentialDTypeMerge.
    # In particular there are hyperedges which could be merged
    # but are missed by this greedy approach.

    ideal_hyperedge_list = [
        Hyperedge(vertices=[0, 4, 7, 8, 11], weight=1),
        Hyperedge(vertices=[1, 4, 7], weight=1),
        Hyperedge(vertices=[1, 6, 10], weight=1),
        Hyperedge(vertices=[1, 8, 11], weight=1),
        Hyperedge(vertices=[1, 9], weight=1),
        Hyperedge(vertices=[2, 9], weight=1),
        Hyperedge(vertices=[3, 5, 6, 10], weight=1),
    ]

    test_network = NISQNetwork(
        server_coupling=[[0, 1], [1, 2], [1, 3]],
        server_qubits={0: [0], 1: [1], 2: [2], 3: [3]}
    )

    test_circuit = Circuit(4)

    test_circuit.add_gate(OpType.CU1, 1.0, [0, 1])
    test_circuit.H(1)
    test_circuit.add_gate(OpType.CU1, 1.0, [1, 3])
    test_circuit.H(1)
    test_circuit.add_gate(OpType.CU1, 1.0, [1, 3])
    test_circuit.add_gate(OpType.CU1, 1.0, [0, 1])
    test_circuit.H(1)
    test_circuit.add_gate(OpType.CU1, 1.0, [0, 1])
    test_circuit.H(1)
    test_circuit.add_gate(OpType.CU1, 1.0, [1, 2])
    test_circuit.add_gate(OpType.CU1, 1.0, [1, 3])
    test_circuit.H(1)
    test_circuit.add_gate(OpType.CU1, 1.0, [0, 1])

    test_hyp_circuit = HypergraphCircuit(test_circuit)

    test_hyp_circuit.vertex_neighbours = {
        i: set() for i in test_hyp_circuit.vertex_list
    }
    test_hyp_circuit.hyperedge_list = []
    test_hyp_circuit.hyperedge_dict = {
        i: [] for i in test_hyp_circuit.vertex_list
    }

    new_hyperedge_list = [
        [0, 4, 7, 8, 11],
        [1, 4, 7],
        [1, 6, 10],
        [1, 8, 11],
        [1, 9],
        [2, 9],
        [3, 5, 6, 10],
    ]

    for new_hyperedge in new_hyperedge_list:
        test_hyp_circuit.add_hyperedge(new_hyperedge)

    test_placement = Placement(
        {
            0: 0,
            1: 1,
            2: 2,
            3: 3,
            4: 0,
            5: 3,
            6: 3,
            7: 0,
            8: 0,
            9: 2,
            10: 3,
            11: 0
        }
    )

    distribution = Distribution(
        circuit=test_hyp_circuit,
        placement=test_placement,
        network=test_network,
    )

    assert distribution.cost() == 7
    assert distribution.circuit.hyperedge_list == ideal_hyperedge_list

    refiner = SequentialDTypeMerge()
    refiner.refine(distribution)

    assert distribution.cost() == 7
    assert distribution.circuit.hyperedge_list == ideal_hyperedge_list


def test_sequential_merge_d_type_intertwined():

    test_network = NISQNetwork(
        server_coupling=[[0, 1], [1, 2]],
        server_qubits={0: [0], 1: [1], 2: [2]}
    )

    test_circuit = Circuit(3)

    test_circuit.add_gate(OpType.CU1, 1.0, [0, 1])
    test_circuit.add_gate(OpType.CU1, 1.0, [1, 2])

    test_circuit.H(1)
    test_circuit.add_gate(OpType.CU1, 1.0, [1, 2])
    test_circuit.H(1)

    test_circuit.add_gate(OpType.CU1, 1.0, [1, 0])

    test_hyp_circuit = HypergraphCircuit(test_circuit)

    test_hyp_circuit.vertex_neighbours = {
        i: set() for i in test_hyp_circuit.vertex_list
    }
    test_hyp_circuit.hyperedge_list = []
    test_hyp_circuit.hyperedge_dict = {
        i: [] for i in test_hyp_circuit.vertex_list
    }

    new_hyperedge_list = [
        [0, 3, 6],
        [1, 3, 6],
        [1, 4],
        [1, 5],
        [2, 4, 5],
    ]

    for new_hyperedge in new_hyperedge_list:
        test_hyp_circuit.add_hyperedge(new_hyperedge)

    test_placement = Placement({0: 0, 1: 1, 2: 2, 3: 2, 4: 2, 5: 2, 6: 0})
    distribution = Distribution(
        circuit=test_hyp_circuit,
        placement=test_placement,
        network=test_network
    )

    assert distribution.cost() == 6

    refiner = SequentialDTypeMerge()
    refiner.refine(distribution)

    assert distribution.cost() == 5

    ideal_hyperedge_list = [
        Hyperedge(vertices=[0, 3, 6], weight=1),
        Hyperedge(vertices=[1, 3, 4, 6], weight=1),
        Hyperedge(vertices=[1, 5], weight=1),
        Hyperedge(vertices=[2, 4, 5], weight=1)
    ]

    assert distribution.circuit.hyperedge_list == ideal_hyperedge_list


def test_sequential_merge_d_type_complex_circuit():

    test_network = NISQNetwork(
        server_coupling=[[0, 1], [1, 2], [1, 3]],
        server_qubits={0: [0], 1: [1], 2: [2], 3: [3]}
    )
    test_circuit = Circuit(4)

    # These gates will be in different hyperedges
    test_circuit.add_gate(OpType.CU1, 1.0, [0, 2])
    test_circuit.Rz(0.3, 0)
    test_circuit.add_gate(OpType.CU1, 1.0, [0, 1])

    # Will be embedded
    test_circuit.H(2)
    test_circuit.add_gate(OpType.CU1, 1.0, [3, 2])
    test_circuit.H(2)

    # Allows for embedding, but will not be
    test_circuit.H(0)
    test_circuit.add_gate(OpType.CU1, 1.0, [0, 2])
    test_circuit.add_gate(OpType.CU1, 1.0, [3, 0])
    test_circuit.H(0)

    test_circuit.add_gate(OpType.CU1, 1.0, [3, 1])
    test_circuit.add_gate(OpType.CU1, 1.0, [1, 0])
    test_circuit.H(3)
    test_circuit.add_gate(OpType.CU1, 1.0, [3, 2])
    test_circuit.H(3)
    test_circuit.add_gate(OpType.CU1, 1.0, [3, 1])

    test_hyp_circuit = HypergraphCircuit(test_circuit)

    test_hyp_circuit.vertex_neighbours = {
        i: set() for i in test_hyp_circuit.vertex_list
    }
    test_hyp_circuit.hyperedge_list = []
    test_hyp_circuit.hyperedge_dict = {
        i: [] for i in test_hyp_circuit.vertex_list
    }

    new_hyperedge_list = [
        [0, 4],
        [0, 5],
        [0, 7, 8],
        [0, 10],
        [1, 5, 9, 10],
        [1, 12],
        [2, 4, 7],
        [2, 6],
        [2, 11],
        [3, 6, 8],
        [3, 9, 12],
        [3, 11],
    ]

    for new_hyperedge in new_hyperedge_list:
        test_hyp_circuit.add_hyperedge(new_hyperedge)

    test_placement = Placement(
        {0: 0, 1: 1, 2: 2, 3: 3, 4: 2, 5: 1, 6: 2,
            7: 2, 8: 1, 9: 3, 10: 0, 11: 2, 12: 3}
    )

    distribution = Distribution(
        circuit=test_hyp_circuit,
        placement=test_placement,
        network=test_network
    )

    assert distribution.cost() == 12

    refiner = SequentialDTypeMerge()
    refiner.refine(distribution)

    assert distribution.cost() == 10

    ideal_hyperedge_list = [
        Hyperedge(vertices=[0, 4, 5], weight=1),
        Hyperedge(vertices=[0, 7, 8], weight=1),
        Hyperedge(vertices=[0, 10], weight=1),
        Hyperedge(vertices=[1, 5, 9, 10, 12], weight=1),
        Hyperedge(vertices=[2, 4, 7, 11], weight=1),
        Hyperedge(vertices=[2, 6], weight=1),
        Hyperedge(vertices=[3, 6, 8, 9, 12], weight=1),
        Hyperedge(vertices=[3, 11], weight=1)
    ]
    assert distribution.circuit.hyperedge_list == ideal_hyperedge_list


def test_sequential_merge_d_type_only_CZ():

    network = NISQNetwork(
        server_coupling=[[0, 1], [1, 2]],
        server_qubits={0: [0], 1: [1], 2: [2]}
    )

    circ = Circuit(3)
    circ.add_gate(OpType.CU1, 1.0, [0, 2])
    circ.add_gate(OpType.CU1, 1.0, [0, 1])
    circ.add_gate(OpType.CU1, 1.0, [0, 2])
    circ.add_gate(OpType.CU1, 1.0, [0, 2])

    hyp_circ = HypergraphCircuit(circ)

    placement = Placement({0: 0, 1: 1, 2: 2, 3: 2, 4: 1, 5: 2, 6: 2})

    old_hyperedge = Hyperedge(vertices=[0, 3, 4, 5, 6])
    new_hyperedge_one = Hyperedge(vertices=[0, 3, 4])
    new_hyperedge_two = Hyperedge(vertices=[0, 5, 6])

    hyp_circ.split_hyperedge(
        old_hyperedge=old_hyperedge,
        new_hyperedge_list=[
            new_hyperedge_one,
            new_hyperedge_two
        ]
    )

    distribution = Distribution(
        circuit=hyp_circ, placement=placement, network=network
    )
    assert distribution.cost() == 4

    refiner = SequentialDTypeMerge()
    refiner.refine(distribution)

    assert distribution.cost() == 2


def test_sequential_merge_d_type_no_new_hyperedges():

    network = NISQNetwork(
        server_coupling=[[0, 1], [1, 2]],
        server_qubits={0: [0], 1: [1], 2: [2]}
    )

    circ = Circuit(3)
    circ.add_gate(OpType.CU1, 1.0, [0, 2])
    circ.add_gate(OpType.CU1, 1.0, [0, 1])
    circ.H(0)
    circ.add_gate(OpType.CU1, 1.0, [0, 2])
    circ.add_gate(OpType.CU1, 1.0, [0, 2])

    hyp_circ = HypergraphCircuit(circ)

    hyperedge_list = [
        Hyperedge(
            vertices=[0, 3, 4],
            weight=1
        ),
        Hyperedge(
            vertices=[0, 5, 6],
            weight=1
        ),
        Hyperedge(
            vertices=[1, 4],
            weight=1
        ),
        Hyperedge(
            vertices=[2, 3, 5, 6],
            weight=1
        )
    ]

    placement = Placement({0: 0, 1: 1, 2: 2, 3: 2, 4: 1, 5: 2, 6: 2})

    distribution = Distribution(
        circuit=hyp_circ, placement=placement, network=network
    )

    assert distribution.circuit.hyperedge_list == hyperedge_list

    refiner = SequentialDTypeMerge()
    refiner.refine(distribution)

    assert distribution.circuit.hyperedge_list == hyperedge_list
