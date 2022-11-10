import pytest
from pytket import Circuit, OpType  # type: ignore
from pytket_dqc import Distribution
from pytket_dqc.circuits import HypergraphCircuit
from pytket_dqc.placement import Placement
from pytket_dqc.utils import check_equivalence
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


def test_vertex_cover_refiner_1():

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

    check_equivalence(
        circ, pytket_circ, distribution.get_qubit_mapping()
    )