from pytket_dqc.networks import NISQNetwork
from pytket import Circuit
from pytket.circuit import OpType  # type: ignore
from pytket_dqc import HypergraphCircuit
from pytket_dqc.placement import Placement
from pytket_dqc.circuits.distribution import Distribution
from pytket_dqc.circuits.hypergraph import Hyperedge
from pytket_dqc.refiners import MergeDType


def test_only_CZ_merge_d_type():

    network = NISQNetwork(server_coupling=[[0, 1], [1, 2]], server_qubits={
                          0: [0], 1: [1], 2: [2]})

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
        circuit=hyp_circ, placement=placement, network=network)
    assert distribution.cost() == 4

    refiner = MergeDType()
    refiner.refine(distribution)

    assert distribution.cost() == 2


def test_no_new_hyperedges():

    network = NISQNetwork(server_coupling=[[0, 1], [1, 2]], server_qubits={
                          0: [0], 1: [1], 2: [2]})

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
        circuit=hyp_circ, placement=placement, network=network)

    assert distribution.circuit.hyperedge_list == hyperedge_list

    refiner = MergeDType()
    refiner.refine(distribution)

    assert distribution.circuit.hyperedge_list == hyperedge_list
