import pickle
import pytest
from pytket import Circuit
from pytket_dqc.placement import Placement
from pytket_dqc.circuits import (
    RegularGraphDistributedCircuit,
    DistributedCircuit,
    Hypergraph,
    Hyperedge,
    BipartiteCircuit,
)

from pytket_dqc.utils.gateset import (
    start_proc,
    end_proc,
    telep_proc,
)
from pytket_dqc.distributors import Brute, Random
from pytket_dqc.networks import NISQNetwork
from pytket.circuit import QControlBox, Op, OpType  # type: ignore

# TODO: Test new circuit classes


def test_hypergraph_is_valid():

    hypgraph = Hypergraph()
    hypgraph.add_vertices([1, 2, 3])
    assert not hypgraph.is_valid()
    hypgraph.add_vertex(0)
    assert hypgraph.is_valid()


# TODO: Include vertex type information in this test
def test_distributed_circuit():

    circ = Circuit(2).CRz(1.0, 0, 1)
    dist_circ = DistributedCircuit(circ)

    assert dist_circ.circuit == circ

    vertex_circuit_map = dist_circ.vertex_circuit_map
    assert vertex_circuit_map[0]["type"] == "qubit"
    assert vertex_circuit_map[1]["type"] == "qubit"
    assert vertex_circuit_map[2]["type"] == "gate"


def test_regular_graph_distributed_circuit():

    dist_circ = RegularGraphDistributedCircuit(3, 2, 1, seed=0)
    network = NISQNetwork([[0, 1], [0, 2]], {0: [0, 1], 1: [2, 3, 4], 2: [5]})
    distributor = Brute()
    placement = distributor.distribute(dist_circ, network)
    cost = placement.cost(dist_circ, network)

    assert cost == 0
    assert placement == Placement({0: 1, 3: 1, 4: 1, 1: 1, 5: 1, 2: 1})


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

    small_circ = Circuit(2).CRz(1.0, 0, 1)
    dist_small_circ = DistributedCircuit(small_circ)

    med_circ = Circuit(4).CRz(1.0, 0, 1).CRz(1.0, 1, 2).CRz(1.0, 2, 3)
    dist_med_circ = DistributedCircuit(med_circ)

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
    circ.CRz(0.3, 1, 0)

    dist_circ = DistributedCircuit(circ)

    assert dist_circ.vertex_list == [0, 2, 1]
    assert dist_circ.hyperedge_list == [
        Hyperedge([0, 2], weight=1),
        Hyperedge([1, 2], weight=1),
    ]

    circ = Circuit(2)
    circ.CRz(1.0, 0, 1)
    circ.CRz(0.3, 1, 0)
    circ.Rz(0.3, 1)
    circ.CRz(1.0, 1, 0)

    dist_circ = DistributedCircuit(circ)

    assert dist_circ.hyperedge_list == [
        Hyperedge([0, 2, 3, 4], weight=1),
        Hyperedge([1, 2, 3], weight=1),
        Hyperedge([1, 4], weight=1),
    ]
    assert dist_circ.vertex_list == [0, 2, 3, 4, 1]

    circ = Circuit(3)
    circ.CRz(0.3, 1, 0)
    circ.CRz(0.3, 0, 1)
    circ.Rz(0.3, 0)
    circ.CRz(0.3, 1, 2)
    circ.H(0)
    circ.CRz(0.3, 1, 0)
    circ.CRz(0.3, 0, 1)

    dist_circ = DistributedCircuit(circ)

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

    dist_circ = DistributedCircuit(circ)

    assert dist_circ.vertex_list == [0, 2, 1]
    assert dist_circ.hyperedge_list == [
        Hyperedge([0, 2], weight=2),
        Hyperedge([1, 2], weight=1),
    ]

    circ = Circuit(2)
    circ.CRz(1.0, 0, 1)
    circ.add_qcontrolbox(cv, [1, 0])
    circ.Rz(0.3, 1)
    circ.CRz(1.0, 1, 0)

    dist_circ = DistributedCircuit(circ)

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

    dist_circ = DistributedCircuit(circ)

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

    circ = Circuit(2).CRz(0.3, 0, 1).H(0).CRz(1.0, 0, 1).CRz(0.3, 1, 0)
    dist_circ = DistributedCircuit(circ)

    placement = Placement({0: 1, 1: 2, 2: 0, 3: 0, 4: 0})

    assert dist_circ.is_placement(placement)

    circ_with_dist = dist_circ.to_pytket_circuit(placement, network)

    test_circ = Circuit()

    server_1 = test_circ.add_q_register("Server 1", 1)
    server_2 = test_circ.add_q_register("Server 2", 1)

    server_0_link_0 = test_circ.add_q_register("Server 0 Link Edge 0", 1)
    server_0_link_1 = test_circ.add_q_register("Server 0 Link Edge 1", 1)
    server_0_link_2 = test_circ.add_q_register("Server 0 Link Edge 2", 1)

    test_circ.add_custom_gate(
        start_proc, [], [server_1[0], server_0_link_0[0]]
    )
    test_circ.add_custom_gate(
        start_proc, [], [server_2[0], server_0_link_2[0]]
    )
    test_circ.CRz(0.3, server_0_link_0[0], server_0_link_2[0])
    test_circ.add_custom_gate(end_proc, [], [server_0_link_0[0], server_1[0]])
    test_circ.H(server_1[0])
    test_circ.add_custom_gate(
        start_proc, [], [server_1[0], server_0_link_1[0]]
    )
    test_circ.CRz(1.0, server_0_link_1[0], server_0_link_2[0])
    test_circ.CRz(0.3, server_0_link_2[0], server_0_link_1[0])
    test_circ.add_custom_gate(end_proc, [], [server_0_link_1[0], server_1[0]])
    test_circ.add_custom_gate(end_proc, [], [server_0_link_2[0], server_2[0]])

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


def test_to_pytket_circuit_detached_gate():
    # This test tests the case where the gate is acted on a server to which
    # the no qubit has been assigned.

    network = NISQNetwork([[0, 1], [1, 2]], {0: [0], 1: [1], 2: [2]})

    circ = Circuit(2).CRz(1.0, 0, 1).H(0).CRz(1.0, 0, 1)
    dist_circ = DistributedCircuit(circ)

    placement = Placement({0: 1, 1: 2, 2: 0, 3: 0})

    assert dist_circ.is_placement(placement)

    circ_with_dist = dist_circ.to_pytket_circuit(placement, network)

    test_circ = Circuit()

    server_1 = test_circ.add_q_register("Server 1", 1)
    server_2 = test_circ.add_q_register("Server 2", 1)

    server_0_link_0 = test_circ.add_q_register("Server 0 Link Edge 0", 1)
    server_0_link_1 = test_circ.add_q_register("Server 0 Link Edge 1", 1)
    server_0_link_2 = test_circ.add_q_register("Server 0 Link Edge 2", 1)

    server_1_link_2 = test_circ.add_q_register("Server 1 Link Edge 2", 1)

    test_circ.add_custom_gate(
        start_proc, [], [server_1[0], server_0_link_0[0]]
    )
    test_circ.add_custom_gate(
        start_proc, [], [server_2[0], server_1_link_2[0]]
    )
    test_circ.add_custom_gate(
        start_proc, [], [server_1_link_2[0], server_0_link_2[0]]
    )
    test_circ.CRz(1.0, server_0_link_0[0], server_0_link_2[0])
    test_circ.add_custom_gate(end_proc, [], [server_0_link_0[0], server_1[0]])
    test_circ.H(server_1[0])
    test_circ.add_custom_gate(
        start_proc, [], [server_1[0], server_0_link_1[0]]
    )
    test_circ.CRz(1.0, server_0_link_1[0], server_0_link_2[0])
    test_circ.add_custom_gate(end_proc, [], [server_0_link_1[0], server_1[0]])
    test_circ.add_custom_gate(
        end_proc, [], [server_0_link_2[0], server_1_link_2[0]]
    )
    test_circ.add_custom_gate(end_proc, [], [server_1_link_2[0], server_2[0]])

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


def test_to_pytket_circuit_gates_on_different_servers():

    network = NISQNetwork([[0, 1], [1, 2]], {0: [0], 1: [1], 2: [2]})

    circ = Circuit(2).CRz(1.0, 0, 1).H(1).CRz(1.0, 0, 1)
    dist_circ = DistributedCircuit(circ)

    placement = Placement({0: 1, 1: 2, 2: 0, 3: 1})

    assert dist_circ.is_placement(placement)

    circ_with_dist = dist_circ.to_pytket_circuit(placement, network)

    test_circ = Circuit()

    server_1 = test_circ.add_q_register("Server 1", 1)
    server_2 = test_circ.add_q_register("Server 2", 1)

    server_0_link_0 = test_circ.add_q_register("Server 0 Link Edge 0", 1)
    server_0_link_1 = test_circ.add_q_register("Server 0 Link Edge 1", 1)
    server_1_link_1 = test_circ.add_q_register("Server 1 Link Edge 1", 1)
    server_1_link_2 = test_circ.add_q_register("Server 1 Link Edge 2", 1)

    test_circ.add_custom_gate(
        start_proc, [], [server_1[0], server_0_link_0[0]]
    )
    test_circ.add_custom_gate(
        start_proc, [], [server_2[0], server_1_link_1[0]]
    )
    test_circ.add_custom_gate(
        start_proc, [], [server_1_link_1[0], server_0_link_1[0]]
    )
    test_circ.CRz(1.0, server_0_link_0[0], server_0_link_1[0])
    test_circ.add_custom_gate(
        end_proc, [], [server_0_link_1[0], server_1_link_1[0]]
    )
    test_circ.add_custom_gate(end_proc, [], [server_1_link_1[0], server_2[0]])
    test_circ.H(server_2[0])
    test_circ.add_custom_gate(
        start_proc, [], [server_2[0], server_1_link_2[0]]
    )
    test_circ.CRz(1.0, server_1[0], server_1_link_2[0])
    test_circ.add_custom_gate(end_proc, [], [server_0_link_0[0], server_1[0]])
    test_circ.add_custom_gate(end_proc, [], [server_1_link_2[0], server_2[0]])

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


def test_to_pytket_circuit_with_branching_distribution_tree():

    network = NISQNetwork(
        [[2, 1], [1, 0], [1, 3], [0, 4]],
        {0: [0], 1: [1], 2: [2], 3: [3], 4: [4]},
    )

    two_CZ_circ = Circuit(3).CRz(1.0, 0, 1).CRz(1.0, 0, 2)
    dist_two_CZ_circ = DistributedCircuit(two_CZ_circ)

    placement_two = Placement({0: 0, 1: 2, 2: 3, 3: 2, 4: 3})
    circ_with_dist = dist_two_CZ_circ.to_pytket_circuit(placement_two, network)

    test_circ = Circuit()

    server_0 = test_circ.add_q_register("Server 0", 1)
    server_2 = test_circ.add_q_register("Server 2", 1)
    server_3 = test_circ.add_q_register("Server 3", 1)

    server_1_link_0 = test_circ.add_q_register("Server 1 Link Edge 0", 1)
    server_2_link_0 = test_circ.add_q_register("Server 2 Link Edge 0", 1)
    server_3_link_0 = test_circ.add_q_register("Server 3 Link Edge 0", 1)

    test_circ.add_custom_gate(
        start_proc, [], [server_0[0], server_1_link_0[0]]
    )
    test_circ.add_custom_gate(
        start_proc, [], [server_1_link_0[0], server_2_link_0[0]]
    )
    test_circ.add_custom_gate(
        start_proc, [], [server_1_link_0[0], server_3_link_0[0]]
    )

    test_circ.CRz(1.0, server_2_link_0[0], server_2[0])
    test_circ.CRz(1.0, server_3_link_0[0], server_3[0])

    test_circ.add_custom_gate(
        end_proc, [], [server_3_link_0[0], server_1_link_0[0]]
    )
    test_circ.add_custom_gate(
        end_proc, [], [server_2_link_0[0], server_1_link_0[0]]
    )
    test_circ.add_custom_gate(end_proc, [], [server_1_link_0[0], server_0[0]])

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


@pytest.mark.skip(reason="Support for teleportation has been disabled")
def test_to_pytket_circuit_with_teleportation():

    network = NISQNetwork(
        [[0, 1], [1, 2], [1, 3]], {0: [0], 1: [1], 2: [2], 3: [3]}
    )

    circ = Circuit(2).CRz(1.0, 0, 1).H(1).CX(1, 0)
    dist_circ = DistributedCircuit(circ)

    placement = Placement({0: 1, 1: 2, 2: 0, 3: 2})
    assert dist_circ.is_placement(placement)

    circ_with_dist = dist_circ.to_pytket_circuit(placement, network)

    test_circ = Circuit()

    server_1 = test_circ.add_q_register("Server 1", 1)
    server_2 = test_circ.add_q_register("Server 2", 1)

    server_0_link_0 = test_circ.add_q_register("Server 0 Link Edge 0", 1)
    server_0_link_2 = test_circ.add_q_register("Server 0 Link Edge 2", 1)
    server_1_link_2 = test_circ.add_q_register("Server 1 Link Edge 2", 1)
    server_2_link_1 = test_circ.add_q_register("Server 2 Link Edge 1", 1)

    test_circ.add_custom_gate(
        start_proc, [], [server_1[0], server_0_link_0[0]]
    )
    test_circ.add_custom_gate(
        start_proc, [], [server_2[0], server_1_link_2[0]]
    )
    test_circ.add_custom_gate(
        start_proc, [], [server_1_link_2[0], server_0_link_2[0]]
    )
    test_circ.CRz(1.0, server_0_link_0[0], server_0_link_2[0])
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


def test_to_relabeled_registers():

    circ = Circuit(3)
    circ.CRz(1.0, 0, 1).H(0).CRz(1.0, 0, 1)
    dist_circ = DistributedCircuit(circ)

    placement = Placement({0: 1, 1: 2, 2: 2, 3: 0, 4: 1})
    assert dist_circ.is_placement(placement)

    circ_with_dist = dist_circ.to_relabeled_registers(placement)

    test_circ = Circuit()
    server_1 = test_circ.add_q_register("Server 1", 1)
    server_2 = test_circ.add_q_register("Server 2", 2)
    test_circ.CRz(1.0, server_1[0], server_2[0]).H(server_1[0]).CRz(
        1.0, server_1[0], server_2[0]
    )

    assert circ_with_dist == test_circ


def test_from_placed_circuit():
    """Tests that generation of BipartiteCircuit.from_placed_circuit() method
    functions as it should.

    Test circuits were manually verified in the packed_examples.ipynb
    example Jupyter Notebook to be correct.
    """
    seed = 27
    distributor = Random()

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
        with open(
            "tests/test_circuits/packing/"
            + f"packed_circuits/packed_circuit{i}.pickle",
            "rb",
        ) as f:
            packed_circuit = pickle.load(f)
        network = NISQNetwork(network_tuple[0], network_tuple[1])

        dist_circ = DistributedCircuit(rebased_circuit)
        placement = distributor.distribute(dist_circ, network, seed=seed)
        bp_circuit = BipartiteCircuit(rebased_circuit, placement)
        assert packed_circuit == bp_circuit.packed_circuit
