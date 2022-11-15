from pytket_dqc.networks import NISQNetwork
from pytket import Circuit
from pytket.circuit import OpType  # type: ignore
from pytket_dqc import HypergraphCircuit
from pytket_dqc.placement import Placement
from pytket_dqc.circuits.distribution import Distribution
from pytket_dqc.circuits.hypergraph import Hyperedge
from pytket_dqc.refiners import SequentialDTypeMerge


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
