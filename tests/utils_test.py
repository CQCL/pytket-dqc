from pytket_dqc.utils import (
    dqc_gateset_predicate,
    DQCPass,
    direct_from_origin,
    ebit_memory_required,
    evicted_gate_count,
)
from pytket import Circuit
from pytket_dqc.circuits import HypergraphCircuit
from pytket_dqc.networks import NISQNetwork
from pytket_dqc.allocators import Brute
from pytket_dqc.placement import Placement
import networkx as nx  # type: ignore
from sympy import Symbol  # type: ignore
import json  # type: ignore
import pytest


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
    with open(
        "tests/test_circuits/pauli_6.json", "r"
    ) as fp:
        circ = Circuit().from_dict(json.load(fp))

    # Comparing against calculation by hand
    assert ebit_memory_required(circ) == {0: 0, 1: 2, 2: 3}

    # Randomly generated circuit of type frac_CZ, depth 6 and 6 qubits
    with open(
        "tests/test_circuits/frac_CZ_6.json", "r"
    ) as fp:
        circ = Circuit().from_dict(json.load(fp))

    # Comparing against calculationby hand
    assert ebit_memory_required(circ) == {0: 0, 1: 4, 2: 0}


def test_evicted_gate_count():
    # This is a randomly generated circuit of type pauli, depth 6 and 6 qubits
    with open(
        "tests/test_circuits/pauli_6.json", "r"
    ) as fp:
        circ = Circuit().from_dict(json.load(fp))
    # Comparing against calculation by hand
    assert evicted_gate_count(circ) == 0

    # Randomly generated circuit of type frac_CZ, depth 6 and 6 qubits
    with open(
        "tests/test_circuits/frac_CZ_6.json", "r"
    ) as fp:
        circ = Circuit().from_dict(json.load(fp))
    # Comparing against calculation by
    assert evicted_gate_count(circ) == 6
