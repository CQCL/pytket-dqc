import pickle  # type: ignore
import json  # type: ignore
import pytest
from pytket import Circuit
from pytket_dqc.placement import Placement
from pytket_dqc.circuits import (
    RegularGraphHypergraphCircuit,
    HypergraphCircuit,
    Hypergraph,
    Hyperedge,
    BipartiteCircuit,
    Distribution,
)
from pytket_dqc.circuits.hypergraph import Vertex

from pytket_dqc.utils.gateset import (
    start_proc,
    end_proc,
    telep_proc,
)
from pytket_dqc.allocators import Brute, Random, HypergraphPartitioning
from pytket_dqc.utils import check_equivalence, DQCPass
from pytket_dqc.networks import NISQNetwork
from pytket.circuit import QControlBox, Op, OpType  # type: ignore

# TODO: Test new circuit classes


def test_embedding_and_not_embedding():

    network = NISQNetwork(
        server_coupling=[[0, 1], [1, 2], [1, 3]],
        server_qubits={0: [0], 1: [1], 2: [2], 3: [3]}
    )

    circ = Circuit(4)

    # These gates will be in different hyperedges
    circ.add_gate(OpType.CU1, 1.0, [0, 2])
    circ.Rz(0.3, 0)
    circ.add_gate(OpType.CU1, 1.0, [0, 1])

    # Will be embedded
    circ.H(2)
    circ.add_gate(OpType.CU1, 1.0, [3, 2])
    circ.H(2)

    # Allows for embedding, but will not be
    circ.H(0)
    circ.add_gate(OpType.CU1, 1.0, [0, 2])
    circ.add_gate(OpType.CU1, 1.0, [3, 0])
    circ.H(0)

    circ.add_gate(OpType.CU1, 1.0, [1, 0])

    hyp_circ = HypergraphCircuit(circ)

    placement = Placement({0: 0, 1: 1, 2: 2, 3: 3, 4: 2,
                          5: 1, 6: 2, 7: 2, 8: 1, 9: 0})

    hyp_circ.vertex_neighbours = {
        0: {4, 5, 7, 8, 9},
        4: {0, 2, 7},
        5: {0, 1, 9},
        7: {0, 2, 4, 8},
        8: {0, 3, 6, 7},
        9: {0, 1, 5},
        1: {5, 9},
        2: {4, 6, 7},
        6: {2, 3, 8},
        3: {8, 6}
    }

    hyp_circ.hyperedge_list = [
        Hyperedge(vertices=[0, 4], weight=1),
        Hyperedge(vertices=[0, 5], weight=1),
        Hyperedge(vertices=[0, 7, 8], weight=1),
        Hyperedge(vertices=[0, 9], weight=1),
        Hyperedge(vertices=[1, 5, 9], weight=1),
        Hyperedge(vertices=[2, 4, 7], weight=1),
        Hyperedge(vertices=[2, 6], weight=1),
        Hyperedge(vertices=[3, 6, 8], weight=1)
    ]

    hyp_circ.hyperedge_dict = {
        0: [
            Hyperedge(vertices=[0, 4], weight=1),
            Hyperedge(vertices=[0, 5], weight=1),
            Hyperedge(vertices=[0, 7, 8], weight=1),
            Hyperedge(vertices=[0, 9], weight=1)
        ],
        4: [
            Hyperedge(vertices=[0, 4], weight=1),
            Hyperedge(vertices=[2, 4, 7], weight=1)
        ],
        5: [
            Hyperedge(vertices=[0, 5], weight=1),
            Hyperedge(vertices=[1, 5, 9], weight=1)
        ],
        7: [
            Hyperedge(vertices=[0, 7, 8], weight=1),
            Hyperedge(vertices=[2, 4, 7], weight=1)
        ],
        8: [
            Hyperedge(vertices=[0, 7, 8], weight=1),
            Hyperedge(vertices=[3, 6, 8], weight=1)
        ],
        9: [
            Hyperedge(vertices=[0, 9], weight=1),
            Hyperedge(vertices=[1, 5, 9], weight=1)
        ],
        1: [
            Hyperedge(vertices=[1, 5, 9], weight=1)
        ],
        2: [
            Hyperedge(vertices=[2, 4, 7], weight=1),
            Hyperedge(vertices=[2, 6], weight=1),
        ],
        6: [
            Hyperedge(vertices=[2, 6], weight=1),
            Hyperedge(vertices=[3, 6, 8], weight=1)
        ],
        3: [
            Hyperedge(vertices=[3, 6, 8], weight=1)
        ]
    }

    distribution = Distribution(
        circuit=hyp_circ, placement=placement, network=network)

    pytket_circ = distribution.to_pytket_circuit()

    check_equivalence(
        circ, pytket_circ, distribution.get_qubit_mapping()
    )


def test_failing_circuit_hyperedge_split_and_merge():

    circ = Circuit(3)
    circ.add_gate(OpType.CU1, 1.0, [0, 1])
    circ.H(0)
    circ.add_gate(OpType.CU1, 1.0, [0, 2])
    hyp_circ = HypergraphCircuit(circ)

    to_merge_hyperedge_one = Hyperedge(vertices=[3, 0], weight=1)
    to_merge_hyperedge_two = Hyperedge(vertices=[0, 4], weight=1)

    # This will fail as [3,0] is not a hyperedge.
    # Note that [0,3] is a hyperedge
    with pytest.raises(Exception) as e_info:
        hyp_circ.merge_hyperedge(
            to_merge_hyperedge_list=[
                to_merge_hyperedge_one,
                to_merge_hyperedge_two
            ]
        )

    assert str(e_info.value) == (
        "At least one hyperedge in to_merge_hyperedge_list " +
        "does not belong to this hypergraph."
    )

    to_merge_hyperedge_one = Hyperedge(vertices=[0, 3], weight=1)
    hyp_circ.merge_hyperedge(
        to_merge_hyperedge_list=[
            to_merge_hyperedge_one,
            to_merge_hyperedge_two
        ]
    )

    assert hyp_circ.hyperedge_list == [
        Hyperedge(vertices=[0, 3, 4], weight=1),
        Hyperedge(vertices=[1, 3], weight=1),
        Hyperedge(vertices=[2, 4], weight=1),
    ]

    old_hyperedge = Hyperedge(vertices=[0, 3, 4], weight=1)
    new_hyperedge_one = Hyperedge(vertices=[3, 0], weight=1)
    new_hyperedge_two = Hyperedge(vertices=[0, 4], weight=1)

    # This will fail as [3,0] is not a valid hyperedge as 0 (the qubit)
    # must come first in the list.
    with pytest.raises(Exception) as e_info:
        hyp_circ.split_hyperedge(
            old_hyperedge=old_hyperedge,
            new_hyperedge_list=[new_hyperedge_one, new_hyperedge_two]
        )

    assert str(
        e_info.value) == (
            "The first element of [3, 0] is required to be a qubit vertex."
    )

    # This has been added to ensure that when an error occurs when adding
    # the hypergraph has not been changed.
    assert hyp_circ.hyperedge_list == [
        Hyperedge(vertices=[0, 3, 4], weight=1),
        Hyperedge(vertices=[1, 3], weight=1),
        Hyperedge(vertices=[2, 4], weight=1),
    ]

    new_hyperedge_one = Hyperedge(vertices=[0, 2, 3], weight=1)

    with pytest.raises(Exception) as e_info:
        hyp_circ.split_hyperedge(
            old_hyperedge=old_hyperedge,
            new_hyperedge_list=[new_hyperedge_one, new_hyperedge_two]
        )

    assert str(
        e_info.value) == (
            "[Hyperedge(vertices=[0, 2, 3], weight=1), " +
            "Hyperedge(vertices=[0, 4], weight=1)] does not match " +
            "the vertices in Hyperedge(vertices=[0, 3, 4], weight=1)"
    )

    assert hyp_circ.hyperedge_list == [
        Hyperedge(vertices=[0, 3, 4], weight=1),
        Hyperedge(vertices=[1, 3], weight=1),
        Hyperedge(vertices=[2, 4], weight=1),
    ]

    new_hyperedge_one = Hyperedge(vertices=[0, 3], weight=1)
    hyp_circ.split_hyperedge(
        old_hyperedge=old_hyperedge,
        new_hyperedge_list=[new_hyperedge_one, new_hyperedge_two]
    )

    assert hyp_circ.hyperedge_list == [
        Hyperedge(vertices=[0, 3], weight=1),
        Hyperedge(vertices=[0, 4], weight=1),
        Hyperedge(vertices=[1, 3], weight=1),
        Hyperedge(vertices=[2, 4], weight=1),
    ]


def test_hypergraph_split_hyperedge():

    hypergraph = Hypergraph()
    hypergraph.add_vertices([Vertex(0), Vertex(1), Vertex(2)])
    hypergraph.add_hyperedge([0, 1, 2])
    hypergraph.add_hyperedge([0, 1])

    old_hyperedge_one = Hyperedge(
        vertices=[Vertex(0), Vertex(1), Vertex(2)], weight=1)
    old_hyperedge_two = Hyperedge(
        vertices=[Vertex(0), Vertex(1)], weight=1)
    new_hyperedge_one = Hyperedge(vertices=[Vertex(0), Vertex(2)], weight=1)
    new_hyperedge_two = Hyperedge(vertices=[Vertex(1), Vertex(2)], weight=1)

    hypergraph.split_hyperedge(
        old_hyperedge=old_hyperedge_one,
        new_hyperedge_list=[new_hyperedge_one, new_hyperedge_two]
    )

    assert hypergraph.vertex_neighbours == {0: {1, 2}, 1: {0, 2}, 2: {0, 1}}
    assert hypergraph.hyperedge_list == [
        new_hyperedge_one,
        new_hyperedge_two,
        old_hyperedge_two
    ]
    assert hypergraph.hyperedge_dict == {
        Vertex(0): [new_hyperedge_one, old_hyperedge_two],
        Vertex(1): [new_hyperedge_two, old_hyperedge_two],
        Vertex(2): [new_hyperedge_one, new_hyperedge_two]
    }

    hypergraph = Hypergraph()
    hypergraph.add_vertices([Vertex(0), Vertex(1), Vertex(2)])
    hypergraph.add_hyperedge([0, 1, 2])

    hypergraph.split_hyperedge(
        old_hyperedge=old_hyperedge_one,
        new_hyperedge_list=[new_hyperedge_one, new_hyperedge_two]
    )

    assert hypergraph.vertex_neighbours == {0: {2}, 1: {2}, 2: {0, 1}}
    assert hypergraph.hyperedge_list == [
        new_hyperedge_one,
        new_hyperedge_two
    ]
    assert hypergraph.hyperedge_dict == {
        Vertex(0): [new_hyperedge_one],
        Vertex(1): [new_hyperedge_two],
        Vertex(2): [new_hyperedge_one, new_hyperedge_two]
    }


def test_hypergraph_merge_hyperedge():

    hypergraph = Hypergraph()
    hypergraph.add_vertices([Vertex(0), Vertex(1), Vertex(2)])
    hypergraph.add_hyperedge([0, 1])
    hypergraph.add_hyperedge([1, 2])
    hypergraph.add_hyperedge([2, 0])

    to_merge_hyperedge_one = Hyperedge(
        vertices=[Vertex(1), Vertex(2)], weight=1
    )
    to_merge_hyperedge_two = Hyperedge(
        vertices=[Vertex(2), Vertex(0)], weight=1
    )
    merged_hyperedge_one = Hyperedge(
        vertices=[Vertex(0), Vertex(1)], weight=1
    )
    merged_hyperedge_two = Hyperedge(
        vertices=[Vertex(1), Vertex(2), Vertex(0)], weight=1
    )

    hypergraph.merge_hyperedge(
        to_merge_hyperedge_list=[
            to_merge_hyperedge_one, to_merge_hyperedge_two
        ]
    )

    assert hypergraph.vertex_neighbours == {
        Vertex(0): {1, 2}, 1: {0, 2}, 2: {0, 1}}
    assert hypergraph.hyperedge_list == [
        merged_hyperedge_one, merged_hyperedge_two
    ]
    assert hypergraph.hyperedge_dict == {
        Vertex(0): [merged_hyperedge_one, merged_hyperedge_two],
        Vertex(1): [merged_hyperedge_one, merged_hyperedge_two],
        Vertex(2): [merged_hyperedge_two]
    }

    hypergraph = Hypergraph()
    hypergraph.add_vertices([0, Vertex(1), Vertex(2)])
    hypergraph.add_hyperedge([1, 2])
    hypergraph.add_hyperedge([0, 1])
    hypergraph.add_hyperedge([2, 0])

    hypergraph.merge_hyperedge(
        to_merge_hyperedge_list=[
            to_merge_hyperedge_one, to_merge_hyperedge_two
        ]
    )

    assert hypergraph.vertex_neighbours == {0: {1, 2}, 1: {0, 2}, 2: {0, 1}}
    assert hypergraph.hyperedge_list == [
        merged_hyperedge_two, merged_hyperedge_one
    ]
    assert hypergraph.hyperedge_dict == {
        Vertex(0): [merged_hyperedge_one, merged_hyperedge_two],
        Vertex(1): [merged_hyperedge_two, merged_hyperedge_one],
        Vertex(2): [merged_hyperedge_two]
    }

    hypergraph = Hypergraph()
    hypergraph.add_vertices([Vertex(0), Vertex(1), Vertex(2)])
    hypergraph.add_hyperedge([1, 2])
    hypergraph.add_hyperedge([2, 0])

    hypergraph.merge_hyperedge(
        to_merge_hyperedge_list=[
            to_merge_hyperedge_one, to_merge_hyperedge_two]
    )

    assert hypergraph.vertex_neighbours == {0: {1, 2}, 1: {0, 2}, 2: {0, 1}}
    assert hypergraph.hyperedge_list == [merged_hyperedge_two]
    assert hypergraph.hyperedge_dict == {
        Vertex(0): [merged_hyperedge_two],
        Vertex(1): [merged_hyperedge_two],
        Vertex(2): [merged_hyperedge_two]
    }


def test_hypergraph_is_valid():

    hypgraph = Hypergraph()
    hypgraph.add_vertices([1, 2, 3])
    assert not hypgraph.is_valid()
    hypgraph.add_vertex(0)
    assert hypgraph.is_valid()


# TODO: Include vertex type information in this test
def test_distributed_circuit():

    circ = Circuit(2).add_gate(OpType.CU1, 1.0, [0, 1])
    dist_circ = HypergraphCircuit(circ)

    assert dist_circ.get_circuit() == circ

    vertex_circuit_map = dist_circ._vertex_circuit_map
    assert vertex_circuit_map[0]["type"] == "qubit"
    assert vertex_circuit_map[1]["type"] == "qubit"
    assert vertex_circuit_map[2]["type"] == "gate"


def test_regular_graph_distributed_circuit():

    circ = RegularGraphHypergraphCircuit(3, 2, 1, seed=0).get_circuit()
    network = NISQNetwork([[0, 1], [0, 2]], {0: [0, 1], 1: [2, 3, 4], 2: [5]})
    allocator = Brute()
    distribution = allocator.allocate(circ, network)
    cost = distribution.cost()

    assert cost == 0
    assert distribution.placement == Placement(
        {0: 1, 3: 1, 4: 1, 1: 1, 5: 1, 2: 1}
    )


def test_hypergraph():

    hypgra = Hypergraph()
    hypgra.add_vertex(0)
    hypgra.add_vertex(1)
    hypgra.add_vertex(2)
    hypgra.add_hyperedge([0, 1])
    hypgra.add_hyperedge([2, 1])
    assert hypgra.vertex_list == [0, 1, 2]
    assert hypgra.hyperedge_list == [
        Hyperedge([0, 1], weight=1),
        Hyperedge([2, 1], weight=1),
    ]


def test_hypergraph_is_placement():

    small_circ = Circuit(2).add_gate(OpType.CU1, 1.0, [0, 1])
    dist_small_circ = HypergraphCircuit(small_circ)

    med_circ = (
        Circuit(4)
        .add_gate(OpType.CU1, 1.0, [0, 1])
        .add_gate(OpType.CU1, 1.0, [1, 2])
        .add_gate(OpType.CU1, 1.0, [2, 3])
    )
    dist_med_circ = HypergraphCircuit(med_circ)

    placement_one = Placement({0: 1, 1: 1, 2: 1, 3: 0, 4: 1, 5: 1, 6: 1})
    assert dist_med_circ.is_placement(placement_one)
    assert not dist_small_circ.is_placement(placement_one)

    placement_two = Placement({0: 2, 1: 2, 2: 2})
    assert not dist_med_circ.is_placement(placement_two)
    assert dist_small_circ.is_placement(placement_two)


def test_hypergrpah_kahypar_hyperedges():

    hypgraph = Hypergraph()

    hypgraph.add_vertices([i + 1 for i in range(6)])
    hypgraph.add_hyperedge([3, 6, 2])
    hypgraph.add_hyperedge([3, 1])
    hypgraph.add_hyperedge([4, 5, 6])

    hyperedge_indices, hyperedges = hypgraph.kahypar_hyperedges()

    assert hyperedge_indices == [0, 3, 5, 8]
    assert hyperedges == [3, 6, 2, 3, 1, 4, 5, 6]


def test_CRz_circuit():
    circ = Circuit(2)
    circ.add_gate(OpType.CU1, 0.3, [1, 0])

    dist_circ = HypergraphCircuit(circ)

    assert dist_circ.vertex_list == [0, 2, 1]
    assert dist_circ.hyperedge_list == [
        Hyperedge([0, 2], weight=1),
        Hyperedge([1, 2], weight=1),
    ]

    circ = Circuit(2)
    circ.add_gate(OpType.CU1, 1.0, [0, 1])
    circ.add_gate(OpType.CU1, 0.3, [1, 0])
    circ.Rz(0.3, 1)
    circ.add_gate(OpType.CU1, 1.0, [1, 0])

    dist_circ = HypergraphCircuit(circ)

    assert dist_circ.hyperedge_list == [
        Hyperedge([0, 2, 3, 4], weight=1),
        Hyperedge([1, 2, 3], weight=1),
        Hyperedge([1, 4], weight=1),
    ]
    assert dist_circ.vertex_list == [0, 2, 3, 4, 1]

    circ = Circuit(3)
    circ.add_gate(OpType.CU1, 0.3, [1, 0])
    circ.add_gate(OpType.CU1, 0.3, [0, 1])
    circ.Rz(0.3, 0)
    circ.add_gate(OpType.CU1, 0.3, [1, 2])
    circ.H(0)
    circ.add_gate(OpType.CU1, 0.3, [1, 0])
    circ.add_gate(OpType.CU1, 0.3, [0, 1])

    dist_circ = HypergraphCircuit(circ)

    assert dist_circ.hyperedge_list == [
        Hyperedge([0, 3, 4], weight=1),
        Hyperedge([0, 6, 7], weight=1),
        Hyperedge([1, 3, 4, 5, 6, 7], weight=1),
        Hyperedge([2, 5], weight=1),
    ]
    assert set(dist_circ.vertex_list) == set([0, 1, 2, 3, 4, 5, 6, 7])


@pytest.mark.skip(reason="QControlBox are not supported for now")
def test_q_control_box_circuits():

    op = Op.create(OpType.V)
    cv = QControlBox(op, 1)

    circ = Circuit(2)
    circ.add_qcontrolbox(cv, [1, 0])

    dist_circ = HypergraphCircuit(circ)

    assert dist_circ.vertex_list == [0, 2, 1]
    assert dist_circ.hyperedge_list == [
        Hyperedge([0, 2], weight=2),
        Hyperedge([1, 2], weight=1),
    ]

    circ = Circuit(2)
    circ.add_gate(OpType.CU1, 1.0, [0, 1])
    circ.add_qcontrolbox(cv, [1, 0])
    circ.Rz(0.3, 1)
    circ.add_gate(OpType.CU1, 1.0, [1, 0])

    dist_circ = HypergraphCircuit(circ)

    assert dist_circ.hyperedge_list == [
        Hyperedge([0, 2], weight=1),
        Hyperedge([0, 3], weight=2),
        Hyperedge([0, 4], weight=1),
        Hyperedge([1, 2, 3], weight=1),
        Hyperedge([1, 4], weight=1),
    ]
    assert dist_circ.vertex_list == [0, 2, 3, 4, 1]

    circ = Circuit(2)
    circ.add_qcontrolbox(cv, [0, 1])
    circ.add_qcontrolbox(cv, [0, 1])
    circ.Rz(0.3, 0)
    circ.add_qcontrolbox(cv, [0, 1])
    circ.H(0)
    circ.add_qcontrolbox(cv, [0, 1])
    circ.add_qcontrolbox(cv, [0, 1])

    dist_circ = HypergraphCircuit(circ)

    assert dist_circ.hyperedge_list == [
        Hyperedge([0, 2, 3], weight=1),
        Hyperedge([0, 4], weight=1),
        Hyperedge([0, 5, 6], weight=1),
        Hyperedge([1, 2], weight=2),
        Hyperedge([1, 3], weight=2),
        Hyperedge([1, 4], weight=2),
        Hyperedge([1, 5], weight=2),
        Hyperedge([1, 6], weight=2),
    ]
    assert dist_circ.vertex_list == [0, 2, 3, 4, 5, 6, 1]


def test_to_pytket_circ_CRz():

    network = NISQNetwork([[0, 1], [1, 2], [0, 2]], {0: [0], 1: [1], 2: [2]})

    circ = (
        Circuit(2)
        .add_gate(OpType.CU1, 0.3, [0, 1])
        .H(0)
        .add_gate(OpType.CU1, 1.0, [0, 1])
        .add_gate(OpType.CU1, 0.3, [1, 0])
    )
    dist_circ = HypergraphCircuit(circ)

    placement = Placement({0: 1, 1: 2, 2: 0, 3: 0, 4: 0})
    distribution = Distribution(dist_circ, placement, network)
    assert distribution.is_valid()

    circ_with_dist = distribution.to_pytket_circuit()

    test_circ = Circuit()

    server_1 = test_circ.add_q_register("server_1", 1)
    server_2 = test_circ.add_q_register("server_2", 1)

    server_0_link = test_circ.add_q_register("server_0_link_register", 2)

    test_circ.add_custom_gate(start_proc, [], [server_1[0], server_0_link[0]])
    test_circ.add_custom_gate(start_proc, [], [server_2[0], server_0_link[1]])
    test_circ.add_gate(OpType.CU1, 0.3, [server_0_link[0], server_0_link[1]])
    test_circ.add_custom_gate(end_proc, [], [server_0_link[0], server_1[0]])
    test_circ.H(server_1[0])
    test_circ.add_custom_gate(start_proc, [], [server_1[0], server_0_link[0]])
    test_circ.add_gate(OpType.CU1, 1.0, [server_0_link[0], server_0_link[1]])
    test_circ.add_gate(OpType.CU1, 0.3, [server_0_link[1], server_0_link[0]])
    test_circ.add_custom_gate(end_proc, [], [server_0_link[0], server_1[0]])
    test_circ.add_custom_gate(end_proc, [], [server_0_link[1], server_2[0]])

    test_circ_command_names = [
        command.op.get_name() for command in test_circ.get_commands()
    ]
    circ_with_dist_command_names = [
        command.op.get_name() for command in circ_with_dist.get_commands()
    ]

    assert test_circ_command_names == circ_with_dist_command_names

    test_circ_command_qubits = [
        command.qubits for command in test_circ.get_commands()
    ]
    circ_with_dist_command_qubits = [
        command.qubits for command in circ_with_dist.get_commands()
    ]

    assert test_circ_command_qubits == circ_with_dist_command_qubits

    assert check_equivalence(
        circ, circ_with_dist, distribution.get_qubit_mapping()
    )


def test_to_pytket_circuit_detached_gate():
    # This test tests the case where the gate is acted on a server to which
    # the no qubit has been assigned.

    network = NISQNetwork([[0, 1], [1, 2]], {0: [0], 1: [1], 2: [2]})

    circ = (
        Circuit(2)
        .add_gate(OpType.CU1, 1.0, [0, 1])
        .H(0)
        .add_gate(OpType.CU1, 1.0, [0, 1])
    )
    dist_circ = HypergraphCircuit(circ)
    placement = Placement({0: 1, 1: 2, 2: 0, 3: 0})
    distribution = Distribution(dist_circ, placement, network)
    assert distribution.is_valid()

    circ_with_dist = distribution.to_pytket_circuit()

    test_circ = Circuit()

    server_1 = test_circ.add_q_register("server_1", 1)
    server_2 = test_circ.add_q_register("server_2", 1)

    server_0_link = test_circ.add_q_register("server_0_link_register", 2)
    server_1_link = test_circ.add_q_register("server_1_link_register", 1)

    test_circ.add_custom_gate(start_proc, [], [server_1[0], server_0_link[0]])
    test_circ.add_custom_gate(start_proc, [], [server_2[0], server_1_link[0]])
    test_circ.add_custom_gate(
        start_proc, [], [server_1_link[0], server_0_link[1]]
    )
    test_circ.add_gate(OpType.CU1, 1.0, [server_0_link[0], server_0_link[1]])
    test_circ.add_custom_gate(end_proc, [], [server_0_link[0], server_1[0]])
    test_circ.add_custom_gate(end_proc, [], [server_1_link[0], server_2[0]])
    test_circ.H(server_1[0])
    test_circ.add_custom_gate(start_proc, [], [server_1[0], server_0_link[0]])
    test_circ.add_gate(OpType.CU1, 1.0, [server_0_link[0], server_0_link[1]])
    test_circ.add_custom_gate(end_proc, [], [server_0_link[0], server_1[0]])
    test_circ.add_custom_gate(end_proc, [], [server_0_link[1], server_2[0]])

    # TODO: Ideally we would compare the circuits directly here, rather than
    # checking the command names. This is prevented by a feature of TKET
    # which required each new box have its own unique ID. This is currently
    # being looked into.

    test_circ_command_names = [
        command.op.get_name() for command in test_circ.get_commands()
    ]
    circ_with_dist_command_names = [
        command.op.get_name() for command in circ_with_dist.get_commands()
    ]

    assert test_circ_command_names == circ_with_dist_command_names

    test_circ_command_qubits = [
        command.qubits for command in test_circ.get_commands()
    ]
    circ_with_dist_command_qubits = [
        command.qubits for command in circ_with_dist.get_commands()
    ]

    assert test_circ_command_qubits == circ_with_dist_command_qubits

    assert test_circ.q_registers == circ_with_dist.q_registers

    assert check_equivalence(
        circ, circ_with_dist, distribution.get_qubit_mapping()
    )


def test_to_pytket_circuit_gates_on_different_servers():

    network = NISQNetwork([[0, 1], [1, 2]], {0: [0], 1: [1], 2: [2]})

    circ = (
        Circuit(2)
        .add_gate(OpType.CU1, 1.0, [0, 1])
        .H(1)
        .add_gate(OpType.CU1, 1.0, [0, 1])
    )
    dist_circ = HypergraphCircuit(circ)

    placement = Placement({0: 1, 1: 2, 2: 0, 3: 1})
    distribution = Distribution(dist_circ, placement, network)
    assert distribution.is_valid()

    circ_with_dist = distribution.to_pytket_circuit()

    test_circ = Circuit()

    server_1 = test_circ.add_q_register("server_1", 1)
    server_2 = test_circ.add_q_register("server_2", 1)

    server_0_link = test_circ.add_q_register("server_0_link_register", 2)
    server_1_link = test_circ.add_q_register("server_1_link_register", 1)

    test_circ.add_custom_gate(start_proc, [], [server_1[0], server_0_link[0]])
    test_circ.add_custom_gate(start_proc, [], [server_2[0], server_1_link[0]])
    test_circ.add_custom_gate(
        start_proc, [], [server_1_link[0], server_0_link[1]]
    )
    test_circ.add_gate(OpType.CU1, 1.0, [server_0_link[0], server_0_link[1]])
    test_circ.add_custom_gate(end_proc, [], [server_1_link[0], server_2[0]])
    test_circ.add_custom_gate(end_proc, [], [server_0_link[1], server_2[0]])
    test_circ.H(server_2[0])
    test_circ.add_custom_gate(start_proc, [], [server_2[0], server_1_link[0]])
    test_circ.add_gate(OpType.CU1, 1.0, [server_1[0], server_1_link[0]])
    test_circ.add_custom_gate(end_proc, [], [server_0_link[0], server_1[0]])
    test_circ.add_custom_gate(end_proc, [], [server_1_link[0], server_2[0]])

    test_circ_command_names = [
        command.op.get_name() for command in test_circ.get_commands()
    ]
    circ_with_dist_command_names = [
        command.op.get_name() for command in circ_with_dist.get_commands()
    ]

    assert test_circ_command_names == circ_with_dist_command_names

    test_circ_command_qubits = [
        command.qubits for command in test_circ.get_commands()
    ]
    circ_with_dist_command_qubits = [
        command.qubits for command in circ_with_dist.get_commands()
    ]

    assert test_circ_command_qubits == circ_with_dist_command_qubits

    assert test_circ.q_registers == circ_with_dist.q_registers

    assert check_equivalence(
        circ, circ_with_dist, distribution.get_qubit_mapping()
    )


def test_to_pytket_circuit_with_branching_distribution_tree():

    network = NISQNetwork(
        [[2, 1], [1, 0], [1, 3], [0, 4]],
        {0: [0], 1: [1], 2: [2], 3: [3], 4: [4]},
    )

    circ = (
        Circuit(3)
        .add_gate(OpType.CU1, 1.0, [0, 1])
        .add_gate(OpType.CU1, 1.0, [0, 2])
    )
    dist_circ = HypergraphCircuit(circ)

    placement = Placement({0: 0, 1: 2, 2: 3, 3: 2, 4: 3})
    distribution = Distribution(dist_circ, placement, network)
    assert distribution.is_valid()

    circ_with_dist = distribution.to_pytket_circuit()

    test_circ = Circuit()

    server_0 = test_circ.add_q_register("server_0", 1)
    server_2 = test_circ.add_q_register("server_2", 1)
    server_3 = test_circ.add_q_register("server_3", 1)

    server_1_link = test_circ.add_q_register("server_1_link_register", 1)
    server_2_link = test_circ.add_q_register("server_2_link_register", 1)
    server_3_link = test_circ.add_q_register("server_3_link_register", 1)

    test_circ.add_custom_gate(start_proc, [], [server_0[0], server_1_link[0]])
    test_circ.add_custom_gate(
        start_proc, [], [server_1_link[0], server_2_link[0]]
    )
    test_circ.add_custom_gate(
        start_proc, [], [server_1_link[0], server_3_link[0]]
    )
    test_circ.add_gate(OpType.CU1, 1.0, [server_2_link[0], server_2[0]])
    test_circ.add_custom_gate(end_proc, [], [server_1_link[0], server_0[0]])
    test_circ.add_gate(OpType.CU1, 1.0, [server_3_link[0], server_3[0]])
    test_circ.add_custom_gate(end_proc, [], [server_2_link[0], server_0[0]])
    test_circ.add_custom_gate(end_proc, [], [server_3_link[0], server_0[0]])

    test_circ_command_names = [
        command.op.get_name() for command in test_circ.get_commands()
    ]
    circ_with_dist_command_names = [
        command.op.get_name() for command in circ_with_dist.get_commands()
    ]

    print(circ_with_dist.get_commands())

    assert test_circ_command_names == circ_with_dist_command_names

    test_circ_command_qubits = [
        command.qubits for command in test_circ.get_commands()
    ]
    circ_with_dist_command_qubits = [
        command.qubits for command in circ_with_dist.get_commands()
    ]

    assert test_circ_command_qubits == circ_with_dist_command_qubits

    assert test_circ.q_registers == circ_with_dist.q_registers

    assert check_equivalence(
        circ, circ_with_dist, distribution.get_qubit_mapping()
    )


def test_to_pytket_circuit_with_embedding_1q():

    network = NISQNetwork(
        [[2, 1], [1, 0], [1, 3], [0, 4]],
        {0: [0], 1: [1], 2: [2], 3: [3], 4: [4, 5]},
    )

    circ = (
        Circuit(3)
        .add_gate(OpType.CU1, 0.3, [0, 1])
        .H(0)
        .Rz(1.0, 0)  # To be embedded
        .H(0)
        .add_gate(OpType.CU1, 0.8, [0, 2])
    )
    hyp_circ = HypergraphCircuit(circ)
    hyp_circ.hyperedge_list = []
    hyp_circ.hyperedge_dict = {v: [] for v in hyp_circ.vertex_list}
    hyp_circ.vertex_neighbours = {v: set() for v in hyp_circ.vertex_list}
    hyp_circ.add_hyperedge([0, 3, 4])  # Hyperedge with embedding
    hyp_circ.add_hyperedge([1, 3])
    hyp_circ.add_hyperedge([2, 4])

    placement = Placement({0: 0, 1: 4, 2: 4, 3: 4, 4: 4})
    distribution = Distribution(hyp_circ, placement, network)
    assert distribution.is_valid()

    circ_with_dist = distribution.to_pytket_circuit()

    assert check_equivalence(
        circ, circ_with_dist, distribution.get_qubit_mapping()
    )


def test_to_pytket_circuit_with_embedding_2q():

    network = NISQNetwork(
        [[2, 1], [1, 0], [1, 3], [0, 4]],
        {0: [0], 1: [1], 2: [2], 3: [3], 4: [4, 5, 6]},
    )

    circ = (
        Circuit(4)
        .add_gate(OpType.CU1, 0.3, [0, 1])
        .H(0)
        .add_gate(OpType.CU1, 1.0, [0, 3])  # To be embedded
        .H(0)
        .add_gate(OpType.CU1, 0.8, [0, 2])
    )
    hyp_circ = HypergraphCircuit(circ)
    hyp_circ.hyperedge_list = []
    hyp_circ.hyperedge_dict = {v: [] for v in hyp_circ.vertex_list}
    hyp_circ.vertex_neighbours = {v: set() for v in hyp_circ.vertex_list}
    hyp_circ.add_hyperedge([0, 4, 6])  # Hyperedge with embedding
    hyp_circ.add_hyperedge([0, 5])
    hyp_circ.add_hyperedge([1, 4])
    hyp_circ.add_hyperedge([2, 6])
    hyp_circ.add_hyperedge([3, 5])

    placement = Placement({0: 0, 1: 4, 2: 4, 3: 4, 4: 4, 5: 3, 6: 4})
    distribution = Distribution(hyp_circ, placement, network)
    assert distribution.is_valid()

    circ_with_dist = distribution.to_pytket_circuit()

    assert check_equivalence(
        circ, circ_with_dist, distribution.get_qubit_mapping()
    )


def test_to_pytket_circuit_circ_with_embeddings():

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

    placement = Placement(
        {
            0: 1,
            1: 1,
            2: 2,
            3: 4,
            4: 1,
            5: 2,
            6: 4,
            7: 4,
            8: 2,
            9: 4,
            10: 0,
            11: 1,
            12: 4,
            13: 3,
        }
    )

    hyp_circ = HypergraphCircuit(circ)
    hyp_circ.hyperedge_list = []
    hyp_circ.hyperedge_dict = {v: [] for v in hyp_circ.vertex_list}
    hyp_circ.vertex_neighbours = {v: set() for v in hyp_circ.vertex_list}

    hyp_circ.add_hyperedge([0, 11, 12])
    hyp_circ.add_hyperedge([0, 5, 7])
    hyp_circ.add_hyperedge([0, 8, 9])
    hyp_circ.add_hyperedge([1, 4, 10, 11, 13])
    hyp_circ.add_hyperedge([2, 4, 5, 6, 13])  # Merged hyperedge
    hyp_circ.add_hyperedge([2, 8, 10])
    hyp_circ.add_hyperedge([3, 6, 7, 9, 12])  # Merged hyperedge

    distribution = Distribution(hyp_circ, placement, network)
    assert distribution.is_valid()

    circ_with_dist = distribution.to_pytket_circuit()

    assert check_equivalence(
        circ, circ_with_dist, distribution.get_qubit_mapping()
    )


def test_to_pytket_circuit_with_hyperedge_requiring_euler():
    # The circuit given below has a hyperedge between the first and last
    # CU1 gates. The gates in between can all be embedded but, to do so,
    # it is required to decompose the middle Hadamard to its Euler form
    # and squash the Rz(0.5) accordingly.

    circ = Circuit(5)
    circ.H(0)
    circ.add_gate(OpType.CU1, [0.3], [0, 1])
    circ.H(0)
    circ.Rz(0.5, 0)
    circ.add_gate(OpType.CU1, [1.0], [0, 3])
    circ.H(0)
    circ.add_gate(OpType.CU1, [1.0], [0, 4])
    circ.Rz(0.5, 0)
    circ.H(0)
    circ.add_gate(OpType.CU1, [0.8], [0, 2])
    circ.Rz(0.3, 0)
    circ.H(0)

    network = NISQNetwork([[0, 1]], {0: [0], 1: [1, 2, 3, 4]},)

    placement = Placement(
        {0: 0, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1}
    )

    hyp_circ = HypergraphCircuit(circ)
    hyp_circ.hyperedge_list = []
    hyp_circ.hyperedge_dict = {v: [] for v in hyp_circ.vertex_list}
    hyp_circ.vertex_neighbours = {v: set() for v in hyp_circ.vertex_list}

    hyp_circ.add_hyperedge([0, 5, 8])  # Merged hyperedge
    hyp_circ.add_hyperedge([0, 6])
    hyp_circ.add_hyperedge([0, 7])
    hyp_circ.add_hyperedge([1, 5])
    hyp_circ.add_hyperedge([2, 8])
    hyp_circ.add_hyperedge([3, 6])
    hyp_circ.add_hyperedge([4, 7])

    distribution = Distribution(hyp_circ, placement, network)
    assert distribution.is_valid()

    assert check_equivalence(
        circ,
        distribution.to_pytket_circuit(),
        distribution.get_qubit_mapping(),
    )


def test_to_pytket_circuit_with_pauli_circ():
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

    allocator = HypergraphPartitioning()
    distribution = allocator.allocate(circ, network, num_rounds=0)

    assert check_equivalence(
        circ,
        distribution.to_pytket_circuit(),
        distribution.get_qubit_mapping(),
    )


def test_to_pytket_circuit_with_random_circ():
    # Randomly generated circuit of type random, depth 6 and 6 qubits
    with open(
        "tests/test_circuits/to_pytket_circuit/random_6.json", "r"
    ) as fp:
        circ = Circuit().from_dict(json.load(fp))

    network = NISQNetwork(
        [[2, 1], [1, 0], [1, 3], [0, 4]],
        {0: [0, 1, 2], 1: [3, 4], 2: [5, 6, 7], 3: [8], 4: [9]},
    )

    allocator = HypergraphPartitioning()
    distribution = allocator.allocate(circ, network, num_rounds=0)

    assert check_equivalence(
        circ,
        distribution.to_pytket_circuit(),
        distribution.get_qubit_mapping(),
    )


def test_to_pytket_circuit_with_frac_cz_circ():
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

    allocator = HypergraphPartitioning()
    distribution = allocator.allocate(circ, network, num_rounds=0)

    assert check_equivalence(
        circ,
        distribution.to_pytket_circuit(),
        distribution.get_qubit_mapping(),
    )


@pytest.mark.skip(reason="Support for teleportation has been disabled")
def test_to_pytket_circuit_with_teleportation():

    network = NISQNetwork(
        [[0, 1], [1, 2], [1, 3]], {0: [0], 1: [1], 2: [2], 3: [3]}
    )

    circ = Circuit(2).add_gate(OpType.CU1, 1.0, [0, 1]).H(1).CX(1, 0)
    dist_circ = HypergraphCircuit(circ)

    placement = Placement({0: 1, 1: 2, 2: 0, 3: 2})
    distribution = Distribution(dist_circ, placement, network)
    assert distribution.is_valid()

    circ_with_dist = distribution.to_pytket_circuit()

    test_circ = Circuit()

    server_1 = test_circ.add_q_register("server_1", 1)
    server_2 = test_circ.add_q_register("server_2", 1)

    server_0_link_0 = test_circ.add_q_register("server_0_link_edge_0", 1)
    server_0_link_2 = test_circ.add_q_register("server_0_link_edge_2", 1)
    server_1_link_2 = test_circ.add_q_register("server_1_link_edge_2", 1)
    server_2_link_1 = test_circ.add_q_register("server_2_link_edge_1", 1)

    test_circ.add_custom_gate(
        start_proc, [], [server_1[0], server_0_link_0[0]]
    )
    test_circ.add_custom_gate(
        start_proc, [], [server_2[0], server_1_link_2[0]]
    )
    test_circ.add_custom_gate(
        start_proc, [], [server_1_link_2[0], server_0_link_2[0]]
    )
    test_circ.add_gate(
        OpType.CU1, 1.0, [server_0_link_0[0], server_0_link_2[0]]
    )
    test_circ.add_custom_gate(end_proc, [], [server_0_link_0[0], server_1[0]])
    test_circ.add_custom_gate(
        end_proc, [], [server_0_link_2[0], server_1_link_2[0]]
    )
    test_circ.add_custom_gate(end_proc, [], [server_1_link_2[0], server_2[0]])
    test_circ.H(server_2[0])
    test_circ.add_custom_gate(
        telep_proc, [], [server_1[0], server_2_link_1[0]]
    )
    test_circ.CX(server_2[0], server_2_link_1[0])
    test_circ.add_custom_gate(
        telep_proc, [], [server_2_link_1[0], server_1[0]]
    )

    test_circ_command_names = [
        command.op.get_name() for command in test_circ.get_commands()
    ]
    circ_with_dist_command_names = [
        command.op.get_name() for command in circ_with_dist.get_commands()
    ]

    assert test_circ_command_names == circ_with_dist_command_names

    test_circ_command_qubits = [
        command.qubits for command in test_circ.get_commands()
    ]
    circ_with_dist_command_qubits = [
        command.qubits for command in circ_with_dist.get_commands()
    ]

    assert test_circ_command_qubits == circ_with_dist_command_qubits

    assert test_circ.q_registers == circ_with_dist.q_registers

    assert check_equivalence(
        circ, circ_with_dist, distribution.get_qubit_mapping()
    )


@pytest.mark.skip(reason="Tests a function that has been removed")
def test_to_relabeled_registers():

    circ = Circuit(3)
    circ.add_gate(OpType.CU1, 1.0, [0, 1]).H(0).add_gate(
        OpType.CU1, 1.0, [0, 1]
    )
    dist_circ = HypergraphCircuit(circ)

    placement = Placement({0: 1, 1: 2, 2: 2, 3: 0, 4: 1})
    assert dist_circ.is_placement(placement)

    circ_with_dist = dist_circ.to_relabeled_registers(placement)

    test_circ = Circuit()
    server_1 = test_circ.add_q_register("server_1", 1)
    server_2 = test_circ.add_q_register("server_2", 2)
    test_circ.add_gate(OpType.CU1, 1.0, [server_1[0], server_2[0]]).H(
        server_1[0]
    ).add_gate(OpType.CU1, 1.0, [server_1[0], server_2[0]])

    assert circ_with_dist == test_circ


def test_from_placed_circuit():
    """Tests that generation of BipartiteCircuit.from_placed_circuit() method
    functions as it should.

    Test circuits were manually verified in the packed_examples.ipynb
    example Jupyter Notebook to be correct.
    """
    seed = 27
    allocator = Random()

    for i in range(6):
        with open(
            f"tests/test_circuits/packing/networks/network{i}.pickle", "rb"
        ) as f:
            network_tuple = pickle.load(f)
        with open(
            "tests/test_circuits/packing/"
            + f"rebased_circuits/rebased_circuit{i}.pickle",
            "rb",
        ) as f:
            rebased_circuit = pickle.load(f)
        DQCPass().apply(rebased_circuit)
        network = NISQNetwork(network_tuple[0], network_tuple[1])
        distribution = allocator.allocate(rebased_circuit, network, seed=seed)
        bp_circuit = BipartiteCircuit(rebased_circuit, distribution.placement)

        with open(
            "tests/test_circuits/packing/"
            + f"qubit_mappings/qubit_mapping{i}.pickle",
            "rb",
        ) as f:
            mapping = pickle.load(f)

        assert check_equivalence(
            rebased_circuit, bp_circuit.packed_circuit, mapping
        )


def test_distribution_initialisation():

    circ = Circuit(3)
    circ.add_gate(OpType.CU1, 1.0, [0, 1]).add_gate(OpType.CU1, 1.0, [0, 2])
    dist_circ = HypergraphCircuit(circ)

    placement = Placement({0: 1, 1: 2, 2: 2, 3: 0, 4: 1})

    network = NISQNetwork(
        [[0, 1], [1, 2]],
        {0: [0, 1, 2], 1: [3, 4, 5], 2: [6, 7, 8]},
    )

    Distribution(dist_circ, placement, network)


def test_get_hyperedge_subcircuit():

    # The test circuit
    circ = Circuit(3)
    circ.add_gate(OpType.CU1, 0.1, [0, 1])  # Gate 3
    circ.Rz(0.2, 0)
    circ.H(0)
    circ.add_gate(OpType.CU1, 0.3, [1, 2])  # Gate 4
    circ.Rz(1, 0)
    circ.add_gate(OpType.CU1, 1.0, [0, 2])  # Gate 5
    circ.H(0)
    circ.add_gate(OpType.CU1, 0.4, [0, 1])  # Gate 6
    hyp_circ = HypergraphCircuit(circ)

    # The hyperedges to test
    hyp_1 = Hyperedge([1, 3, 4, 6])  # This one is in hyp_circ
    hyp_2 = Hyperedge([0, 3, 5])  # This is a merge of two (has embeddings)

    # Testing for hyp_1
    test_c = Circuit(3)
    test_c.add_gate(OpType.CU1, 0.1, [0, 1])
    test_c.add_gate(OpType.CU1, 0.3, [1, 2])
    test_c.add_gate(OpType.CU1, 0.4, [0, 1])

    assert test_c.get_commands() == hyp_circ.get_hyperedge_subcircuit(hyp_1)

    # Testing for hyp_2
    test_c = Circuit(3)
    test_c.add_gate(OpType.CU1, 0.1, [0, 1])
    test_c.Rz(0.2, 0)
    test_c.H(0)
    test_c.Rz(1, 0)
    test_c.add_gate(OpType.CU1, 1.0, [0, 2])

    assert test_c.get_commands() == hyp_circ.get_hyperedge_subcircuit(hyp_2)


def test_requires_h_embedded_cu1():

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

    # The initial hyperedges do not have embeddings
    hyp_circ = HypergraphCircuit(circ)
    for hyperedge in hyp_circ.hyperedge_list:
        assert not hyp_circ.requires_h_embedded_cu1(hyperedge)

    # Consider some merged hyperedges
    hyp_0 = Hyperedge([0, 5, 7, 11, 12])
    hyp_2 = Hyperedge([2, 4, 5, 6, 13])
    hyp_3 = Hyperedge([3, 6, 7, 9, 12])
    assert hyp_circ.requires_h_embedded_cu1(hyp_0)
    assert hyp_circ.requires_h_embedded_cu1(hyp_2)
    assert not hyp_circ.requires_h_embedded_cu1(hyp_3)
