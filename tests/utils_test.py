from pytket_dqc.utils import (
    dqc_gateset_predicate,
    DQCPass,
    direct_from_origin,
    ebit_memory_required,
    evicted_gate_count,
    check_equivalence,
)
from pytket import Circuit, OpType  # type: ignore
from pytket.pauli import Pauli  # type: ignore
from pytket_dqc.circuits import HypergraphCircuit
from pytket_dqc.networks import NISQNetwork
from pytket_dqc.allocators import Brute
from pytket_dqc.placement import Placement
import networkx as nx  # type: ignore
from sympy import Symbol  # type: ignore
import json  # type: ignore
import pickle  # type: ignore
from pytket.circuit import PauliExpBox  # type: ignore
from pytket.passes import DecomposeBoxes  # type: ignore
import numpy as np  # type: ignore
import pytest  # type: ignore
from pytket_dqc.utils.qasm import to_qasm_str
from pytket.qasm import circuit_from_qasm_str


def test_qasm():

    network = NISQNetwork([[0, 1], [0, 2]], {0: [0], 1: [1], 2: [2]})

    circ = Circuit(2)
    circ.add_gate(OpType.CU1, 1.0, [0, 1]).H(0).Rz(
        0.3, 0).H(0).add_gate(OpType.CU1, 1.0, [0, 1])
    dist_circ = HypergraphCircuit(circ)

    placement = Placement({0: 1, 1: 2, 2: 0, 3: 0})
    assert dist_circ.is_placement(placement)

    circ_with_dist = dist_circ.to_pytket_circuit(placement, network)
    qasm_str = to_qasm_str(circ_with_dist)

    qasm_circ = circuit_from_qasm_str(qasm_str)

    assert qasm_circ == circ_with_dist


def test_rebase():

    circ = Circuit(2).CY(0, 1)
    assert not (dqc_gateset_predicate.verify(circ))
    DQCPass().apply(circ)
    assert dqc_gateset_predicate.verify(circ)


@pytest.mark.skip(reason="Support for CX gates temporarily disabled")
def test_CX_circuit():

    circ = Circuit(3).CX(0, 1).CZ(1, 2).H(1).CX(1, 0)
    assert dqc_gateset_predicate.verify(circ)

    dist_circ = HypergraphCircuit(circ)

    network = NISQNetwork([[0, 1]], {0: [0, 1], 1: [2, 3]})

    allocator = Brute()
    placement = allocator.allocate(dist_circ, network)

    assert placement == Placement({0: 0, 3: 0, 5: 0, 1: 0, 4: 0, 2: 1})
    assert placement.cost(dist_circ, network) == 1

    placement = Placement({0: 0, 1: 1, 2: 1, 3: 0, 4: 1, 5: 0})
    assert placement.cost(dist_circ, network) == 3

    # TODO: Add a circuit generation test here when both branches
    # have been merged


def test_direct_from_origin():

    G = nx.Graph()
    edges = [(0, 1), (1, 2), (1, 3), (0, 4), (2, 5), (2, 7), (1, 6)]
    G.add_edges_from(edges)

    from_one_ideal = [(1, 0), (1, 2), (1, 3), (1, 6), (0, 4), (2, 5), (2, 7)]
    from_two_ideal = [(2, 1), (2, 5), (2, 7), (1, 0), (1, 3), (1, 6), (0, 4)]

    assert direct_from_origin(G, 1) == from_one_ideal
    assert direct_from_origin(G, 2) == from_two_ideal

    G_reordered = nx.Graph()
    G_reordered.add_edges_from(
        [(3, 1), (1, 2), (0, 4), (1, 0), (5, 2), (1, 6), (2, 7)]
    )

    assert direct_from_origin(G_reordered, 1) == from_one_ideal
    assert direct_from_origin(G_reordered, 2) == from_two_ideal


def test_symbolic_circuit():

    a = Symbol("alpha")
    circ = Circuit(1)
    circ.Rx(a, 0)

    assert not dqc_gateset_predicate.verify(circ)


def test_ebit_memory_required():
    # This is a randomly generated circuit of type pauli, depth 6 and 6 qubits
    with open("tests/test_circuits/pauli_6.json", "r") as fp:
        circ = Circuit().from_dict(json.load(fp))

    # Comparing against calculation by hand
    assert ebit_memory_required(circ) == {0: 0, 1: 2, 2: 3}

    # Randomly generated circuit of type frac_CZ, depth 6 and 6 qubits
    with open("tests/test_circuits/frac_CZ_6.json", "r") as fp:
        circ = Circuit().from_dict(json.load(fp))

    # Comparing against calculationby hand
    assert ebit_memory_required(circ) == {0: 0, 1: 4, 2: 0}


def test_evicted_gate_count():
    # This is a randomly generated circuit of type pauli, depth 6 and 6 qubits
    with open("tests/test_circuits/pauli_6.json", "r") as fp:
        circ = Circuit().from_dict(json.load(fp))
    # Comparing against calculation by hand
    assert evicted_gate_count(circ) == 0

    # Randomly generated circuit of type frac_CZ, depth 6 and 6 qubits
    with open("tests/test_circuits/frac_CZ_6.json", "r") as fp:
        circ = Circuit().from_dict(json.load(fp))
    # Comparing against calculation by
    assert evicted_gate_count(circ) == 6


def test_verification_from_placed_circuit():
    # This is the same test as in ``test_from_placed_circuit`` but instead of
    # distributing the ``rebased_circuit`` we are just verifying that
    # ``rebased_circuit`` and ``packed_circuit`` are actually equivalent.
    # This is meant to be a test for ``check_equivalence`` in utils.

    rebased_circuits = dict()
    packed_circuits = dict()
    qubit_mappings = dict()

    for i in range(6):
        with open(
            "tests/test_circuits/packing/"
            + f"rebased_circuits/rebased_circuit{i}.pickle",
            "rb",
        ) as f:
            rebased_circuits[i] = pickle.load(f)
        with open(
            "tests/test_circuits/packing/"
            + f"packed_circuits/packed_circuit{i}.pickle",
            "rb",
        ) as f:
            packed_circuits[i] = pickle.load(f)

    qubit_mappings[0] = {
        rebased_circuits[0].qubits[0]: packed_circuits[0].qubits[1],
        rebased_circuits[0].qubits[1]: packed_circuits[0].qubits[0],
    }
    qubit_mappings[1] = {
        rebased_circuits[1].qubits[0]: packed_circuits[1].qubits[2],
        rebased_circuits[1].qubits[1]: packed_circuits[1].qubits[0],
    }
    qubit_mappings[2] = {
        rebased_circuits[2].qubits[0]: packed_circuits[2].qubits[2],
        rebased_circuits[2].qubits[1]: packed_circuits[2].qubits[0],
    }
    qubit_mappings[3] = {
        rebased_circuits[3].qubits[0]: packed_circuits[3].qubits[3],
        rebased_circuits[3].qubits[1]: packed_circuits[3].qubits[0],
    }
    qubit_mappings[4] = {
        rebased_circuits[4].qubits[0]: packed_circuits[4].qubits[5],
        rebased_circuits[4].qubits[1]: packed_circuits[4].qubits[3],
        rebased_circuits[4].qubits[2]: packed_circuits[4].qubits[0],
    }
    qubit_mappings[5] = {
        rebased_circuits[5].qubits[0]: packed_circuits[5].qubits[6],
        rebased_circuits[5].qubits[1]: packed_circuits[5].qubits[7],
        rebased_circuits[5].qubits[2]: packed_circuits[5].qubits[8],
        rebased_circuits[5].qubits[3]: packed_circuits[5].qubits[0],
        rebased_circuits[5].qubits[4]: packed_circuits[5].qubits[1],
        rebased_circuits[5].qubits[5]: packed_circuits[5].qubits[2],
    }

    for i in range(6):
        assert check_equivalence(
            rebased_circuits[i], packed_circuits[i], qubit_mappings[i]
        )


def test_verification_rebase_simple():
    c = Circuit(2).CZ(0, 1)
    orig_c = c.copy()
    DQCPass().apply(c)

    # Check equivalence of unitaries explicitly
    assert np.allclose(orig_c.get_unitary(), c.get_unitary())
    # Check equivalence via PyZX
    assert check_equivalence(orig_c, c, {q: q for q in c.qubits})


def test_verification_rebase_random():
    # Creates a random circuit, rebases it and uses ``check_equivalence``
    # to verify they are equivalent
    np.random.seed(42)
    n_qubits = 8
    depth = 8

    c = Circuit(n_qubits)

    qubit_list = [i for i in range(n_qubits)]
    pauli_list = [Pauli.X, Pauli.Y, Pauli.X, Pauli.I]

    for _ in range(depth):

        # Randomly reorder the qubits on which the gate will act, generate
        # random angle, and choose random Pauli string.
        subset = np.random.permutation(qubit_list)
        angle = np.random.uniform(-2, 2)
        random_pauli = np.random.choice(pauli_list, n_qubits)

        # Generate gate corresponding to pauli string and angle
        pauli_box = PauliExpBox(random_pauli, angle)
        c.add_pauliexpbox(pauli_box, subset)

    DecomposeBoxes().apply(c)
    orig_c = c.copy()
    DQCPass().apply(c)

    # Check equivalence of unitaries explicitly
    assert np.allclose(orig_c.get_unitary(), c.get_unitary())
    # Check equivalence via PyZX
    assert check_equivalence(orig_c, c, {q: q for q in c.qubits})


def test_tk2_to_cu1():

    np.random.seed(42)
    a = round(np.random.uniform(-2, 2), 2)
    b = round(np.random.uniform(-2, 2), 2)
    c = round(np.random.uniform(-2, 2), 2)

    circ = Circuit(2).add_gate(OpType.TK2, [a, b, c], [0, 1])
    orig_circ = circ.copy()
    DQCPass().apply(circ)
    assert dqc_gateset_predicate.verify(circ)
    # Check equivalence of unitaries explicitly
    assert np.allclose(orig_circ.get_unitary(), circ.get_unitary())
    # Check equivalence via PyZX
    assert check_equivalence(orig_circ, circ, {q: q for q in circ.qubits})


def test_tk1_to_euler():

    np.random.seed(42)
    a = round(np.random.uniform(-2, 2), 2)
    rnd_b = round(np.random.uniform(-2, 2), 2)
    c = round(np.random.uniform(-2, 2), 2)

    b_values = [rnd_b, 0, 1, 2, 0.5, 1.5, -0.5, 3.5]

    for b in b_values:
        circ = Circuit(1).add_gate(OpType.TK1, [a, b, c], [0])
        orig_circ = circ.copy()
        DQCPass().apply(circ)
        assert dqc_gateset_predicate.verify(circ)
        # Check equivalence of unitaries explicitly
        assert np.allclose(orig_circ.get_unitary(), circ.get_unitary())
        # Check equivalence via PyZX
        assert check_equivalence(orig_circ, circ, {q: q for q in circ.qubits})


def test_verify_non_equal():

    h_circ = Circuit(1).H(0)
    s_circ = Circuit(1).S(0)
    assert not check_equivalence(h_circ, s_circ, {q: q for q in h_circ.qubits})

    ab_circ = Circuit(2).CX(0, 1).CX(1, 0)
    ba_circ = Circuit(2).CX(1, 0).CX(0, 1)
    # Check inequivalence of unitaries explicitly
    assert not np.allclose(ab_circ.get_unitary(), ba_circ.get_unitary())
    # Check inequivalence via PyZX
    assert not check_equivalence(
        ab_circ, ba_circ, {q: q for q in ab_circ.qubits}
    )
