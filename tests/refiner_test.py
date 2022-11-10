import pytest
import json  # type: ignore
from pytket import Circuit, OpType  # type: ignore
from pytket_dqc import Distribution
from pytket_dqc.circuits import HypergraphCircuit
from pytket_dqc.placement import Placement
from pytket_dqc.utils import check_equivalence, DQCPass
from pytket_dqc.networks import NISQNetwork
from pytket_dqc.refiners.vertex_cover import (
    VertexCover,
    get_min_covers,
)


def test_all_covers():
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