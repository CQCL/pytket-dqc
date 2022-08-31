from pytket_dqc.utils import (
    dqc_gateset_predicate,
    dqc_rebase,
    direct_from_origin,
    ebit_memory_required,
    evicted_gate_count,
)
from pytket import Circuit
from pytket_dqc.circuits import DistributedCircuit
from pytket_dqc.networks import NISQNetwork, AllToAll
from pytket_dqc.distributors import Brute
from pytket_dqc.placement import Placement
import networkx as nx  # type: ignore
from sympy import Symbol  # type: ignore
import json  # type: ignore
import pytest


def test_rebase():

    circ = Circuit(2).CY(0, 1)
    assert not (dqc_gateset_predicate.verify(circ))
    dqc_rebase.apply(circ)
    assert dqc_gateset_predicate.verify(circ)


@pytest.mark.skip(reason="Support for CX gates temporarily disabled")
def test_CX_circuit():

    circ = Circuit(3).CX(0, 1).CZ(1, 2).H(1).CX(1, 0)
    assert dqc_gateset_predicate.verify(circ)

    dist_circ = DistributedCircuit(circ)

    network = NISQNetwork([[0, 1]], {0: [0, 1], 1: [2, 3]})

    distributor = Brute()
    placement = distributor.distribute(dist_circ, network)

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
    network = AllToAll(3, 2)

    # This is a randomly generated circuit of type pauli, depth 6 and 6 qubits
    with open(
        "tests/test_circuits/2f1cc964-1518-4109-97ca-5538906a3dff.json", "r"
    ) as fp:
        circ = Circuit().from_dict(json.load(fp))
        dqc_rebase.apply(circ)
        pauli_circ = DistributedCircuit(circ)
    placement = Placement(
        {
            0: 2,
            1: 2,
            2: 1,
            3: 1,
            4: 0,
            5: 0,
            6: 2,
            7: 0,
            8: 1,
            9: 2,
            10: 2,
            11: 1,
            12: 0,
            13: 0,
            14: 1,
            15: 2,
            16: 2,
            17: 2,
            18: 1,
            19: 0,
            20: 1,
            21: 1,
            22: 2,
            23: 2,
            24: 2,
            25: 2,
            26: 1,
            27: 1,
            28: 2,
            29: 2,
            30: 2,
            31: 2,
            32: 0,
            33: 2,
            34: 2,
            35: 2,
            36: 0,
            37: 2,
            38: 2,
            39: 2,
        }
    )
    pauli_final = pauli_circ.to_pytket_circuit(placement, network)
    # Comparing against calculation by hand
    assert ebit_memory_required(pauli_final) == {0: 0, 1: 2, 2: 3}

    # Randomly generated circuit of type frac_CZ=0.7, depth 6 and 6 qubits
    with open(
        "tests/test_circuits/f9f22168-8168-48ad-baed-aceb2c9aca4d.json", "r"
    ) as fp:
        circ = Circuit().from_dict(json.load(fp))
        dqc_rebase.apply(circ)
        frac_CZ_circ = DistributedCircuit(circ)
    placement = Placement(
        {
            0: 1,
            1: 2,
            2: 0,
            3: 1,
            4: 0,
            5: 2,
            6: 1,
            7: 1,
            8: 1,
            9: 1,
            10: 2,
            11: 0,
            12: 1,
            13: 1,
            14: 1,
            15: 1,
            16: 2,
            17: 1,
            18: 1,
            19: 1,
            20: 1,
            21: 1,
            22: 1,
            23: 1,
        }
    )
    frac_CZ_final = frac_CZ_circ.to_pytket_circuit(placement, network)
    # Comparing against calculationby hand
    assert ebit_memory_required(frac_CZ_final) == {0: 0, 1: 4, 2: 0}


def test_evicted_gate_count():
    network = AllToAll(3, 2)

    # This is a randomly generated circuit of type pauli, depth 6 and 6 qubits
    with open(
        "tests/test_circuits/2f1cc964-1518-4109-97ca-5538906a3dff.json", "r"
    ) as fp:
        circ = Circuit().from_dict(json.load(fp))
        dqc_rebase.apply(circ)
        pauli_circ = DistributedCircuit(circ)
    placement = Placement(
        {
            0: 2,
            1: 2,
            2: 1,
            3: 1,
            4: 0,
            5: 0,
            6: 2,
            7: 0,
            8: 1,
            9: 2,
            10: 2,
            11: 1,
            12: 0,
            13: 0,
            14: 1,
            15: 2,
            16: 2,
            17: 2,
            18: 1,
            19: 0,
            20: 1,
            21: 1,
            22: 2,
            23: 2,
            24: 2,
            25: 2,
            26: 1,
            27: 1,
            28: 2,
            29: 2,
            30: 2,
            31: 2,
            32: 0,
            33: 2,
            34: 2,
            35: 2,
            36: 0,
            37: 2,
            38: 2,
            39: 2,
        }
    )
    pauli_final = pauli_circ.to_pytket_circuit(placement, network)
    # Comparing against calculationby hand
    assert evicted_gate_count(pauli_final) == 0

    # Randomly generated circuit of type frac_CZ=0.7, depth 6 and 6 qubits
    with open(
        "tests/test_circuits/f9f22168-8168-48ad-baed-aceb2c9aca4d.json", "r"
    ) as fp:
        circ = Circuit().from_dict(json.load(fp))
        dqc_rebase.apply(circ)
        frac_CZ_circ = DistributedCircuit(circ)
    placement = Placement(
        {
            0: 1,
            1: 2,
            2: 0,
            3: 1,
            4: 0,
            5: 2,
            6: 1,
            7: 1,
            8: 1,
            9: 1,
            10: 2,
            11: 0,
            12: 1,
            13: 1,
            14: 1,
            15: 1,
            16: 2,
            17: 1,
            18: 1,
            19: 1,
            20: 1,
            21: 1,
            22: 1,
            23: 1,
        }
    )
    frac_CZ_final = frac_CZ_circ.to_pytket_circuit(placement, network)
    # Comparing against calculationby hand
    assert evicted_gate_count(frac_CZ_final) == 6
