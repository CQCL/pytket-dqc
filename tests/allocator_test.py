from pytket_dqc.allocators import (
    Annealing,
    Random,
    Ordered,
    Routing,
    HypergraphPartitioning,
    Brute,
)
from pytket_dqc.allocators.annealing import acceptance_criterion
from pytket_dqc import HypergraphCircuit, Distribution
from pytket import Circuit
from pytket_dqc.networks import NISQNetwork
from pytket_dqc.allocators.ordered import order_reducing_size
from pytket_dqc.placement import Placement
import kahypar as kahypar  # type:ignore
from pytket.circuit import QControlBox, Op, OpType  # type:ignore
import importlib_resources
import pytest
import json


# TODO: Test that the placement returned by routing does not
# have any cost two edges


def test_annealing_distribute():

    network = NISQNetwork([[0, 1], [0, 2]], {0: [0], 1: [1, 2], 2: [3, 4]})

    circ = (
        Circuit(4)
        .add_gate(OpType.CU1, 1.0, [0, 3])
        .H(3)
        .add_gate(OpType.CU1, 1.0, [1, 3])
        .H(3)
        .add_gate(OpType.CU1, 1.0, [2, 3])
        .H(3)
    )

    allocator = Annealing()

    distribution = allocator.allocate(
        circ,
        network,
        seed=2,
        iterations=1,
        initial_place_method=Ordered(),
    )

    assert distribution.placement == Placement(
        {0: 1, 1: 1, 2: 2, 3: 0, 4: 1, 5: 1, 6: 1}
    )
    assert distribution.cost() == 5

    distribution = allocator.allocate(
        circ,
        network,
        seed=1,
        iterations=1,
        initial_place_method=Ordered(),
    )

    assert distribution.placement == Placement(
        {0: 1, 1: 1, 2: 2, 3: 2, 4: 0, 5: 1, 6: 1}
    )
    assert distribution.cost() == 8


def test_acceptance_criterion():

    assert acceptance_criterion(1, 10) >= 1
    assert acceptance_criterion(-1, 10) < 1


def test_graph_initial_partitioning():

    network = NISQNetwork([[0, 1], [0, 2]], {0: [0], 1: [1, 2], 2: [3, 4]})

    circ = (
        Circuit(4)
        .add_gate(OpType.CU1, 1.0, [0, 3])
        .H(3)
        .add_gate(OpType.CU1, 1.0, [1, 3])
        .H(3)
        .add_gate(OpType.CU1, 1.0, [2, 3])
        .H(3)
    )

    allocator = HypergraphPartitioning()

    initial_distribution = allocator.allocate(
        circ, network, seed=1
    )

    assert initial_distribution.placement == Placement(
        {0: 2, 1: 0, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1}
    ) or initial_distribution.placement == Placement(
        {0: 2, 1: 1, 2: 2, 3: 1, 4: 1, 5: 1, 6: 1}
    )


def test_graph_partitioning_make_valid():

    network = NISQNetwork([[0, 1], [0, 2]], {0: [0], 1: [1, 2], 2: [3, 4, 5]})

    circ = (
        Circuit(4)
        .add_gate(OpType.CU1, 1.0, [0, 3])
        .H(0)
        .add_gate(OpType.CU1, 1.0, [1, 3])
        .H(1)
        .add_gate(OpType.CU1, 1.0, [2, 3])
        .H(2)
        .add_gate(OpType.CU1, 1.0, [0, 3])
        .add_gate(OpType.CU1, 1.0, [1, 3])
        .add_gate(OpType.CU1, 1.0, [2, 3])
    )
    dist_circ = HypergraphCircuit(circ)

    allocator = HypergraphPartitioning()
    bad_placement = Placement({v: 2 for v in dist_circ.vertex_list})
    assert not bad_placement.is_valid(dist_circ, network)

    distribution = Distribution(
        dist_circ, bad_placement, network
    )
    allocator.make_valid(distribution, seed=1)
    good_placement = Placement(
        {0: 0, 1: 2, 2: 2, 3: 2, 4: 2, 5: 2, 6: 2, 7: 2, 8: 2, 9: 2}
    )

    assert distribution.is_valid()
    assert distribution.placement == good_placement


@pytest.mark.skip(reason="Circuit contains CX gates that are not supported.")
# NOTE: Moreover, when we do support them again (after including Junyi's work)
# the hypergraph will likely be different, so there is no guarantee the inital
# solution by KaHyPar will fail to be valid. Perhaps we should just remove
# this test, since validity is asserted at the end of refinement anyway.
def test_refinement_makes_valid():
    """In the case of an initial partition using KaHyPar this test fails
    since KaHyPar returns an invalid placement. Refinement fixes this.
    """
    server_coupling = [[0, 1], [1, 2]]
    server_qubits = {
        0: [0, 1],
        1: [2],
        2: [3, 4, 5],
    }
    network = NISQNetwork(server_coupling, server_qubits)

    with open("tests/test_circuits/not_valid_circ.json", "r") as fp:
        circuit = Circuit().from_dict(json.load(fp))

    allocator = HypergraphPartitioning()

    distribution = allocator.allocate(circuit, network, seed=0)
    assert distribution.is_valid()


def test_graph_partitioning_unused_qubits():

    network = NISQNetwork([[0, 1], [0, 2]], {0: [0], 1: [1, 2], 2: [3, 4]})
    allocator = HypergraphPartitioning()

    circ = Circuit(2)

    distribution = allocator.allocate(circ, network, seed=1)
    assert distribution.placement == Placement({0: 1, 1: 0})

    circ = Circuit(0)

    distribution = allocator.allocate(circ, network, seed=1)
    assert distribution.placement == Placement(dict())

    circ = Circuit(3).add_gate(OpType.CU1, 1.0, [1, 2])

    distribution = allocator.allocate(circ, network, seed=1)
    assert (distribution.placement == Placement({0: 2, 1: 1, 2: 1, 3: 1})) or (
        distribution.placement == Placement({0: 1, 1: 2, 2: 2, 3: 2})
    )


def test_kahypar_install():

    num_nodes = 4
    num_nets = 3

    hyperedge_indices = [0, 2, 4, 6]
    hyperedges = [0, 1, 1, 2, 2, 3]

    k = 2

    hypergraph = kahypar.Hypergraph(
        num_nodes, num_nets, hyperedge_indices, hyperedges, k
    )

    context = kahypar.Context()

    package_path = importlib_resources.files("pytket_dqc")
    ini_path = f"{package_path}/allocators/km1_kKaHyPar_sea20.ini"
    context.loadINIconfiguration(ini_path)

    context.setK(k)
    context.setEpsilon(0.03)
    context.suppressOutput(True)

    kahypar.partition(hypergraph, context)
    placement = [hypergraph.blockID(i) for i in range(hypergraph.numNodes())]

    assert placement == [0, 0, 1, 1] or placement == [1, 1, 0, 0]


def test_order_reducing_size():
    my_dict = {0: [0, 1], 2: [5, 6, 7, 8], 1: [2, 3, 4]}
    assert order_reducing_size(my_dict) == {
        2: [5, 6, 7, 8],
        1: [2, 3, 4],
        0: [0, 1],
    }


def test_random_allocator():

    circ = (
        Circuit(3)
        .add_gate(OpType.CU1, 1.0, [0, 2])
        .add_gate(OpType.CU1, 1.0, [1, 2])
    )

    network = NISQNetwork([[0, 1], [0, 2]], {0: [0, 1], 1: [2, 3], 2: [4]})

    allocator = Random()
    distribution = allocator.allocate(circ, network, seed=0)

    assert distribution.placement == Placement({0: 1, 3: 1, 1: 0, 4: 1, 2: 2})
    assert distribution.cost() == 3


def test_ordered_allocator():

    small_circ = Circuit(2).add_gate(OpType.CU1, 1.0, [0, 1])

    med_circ = (
        Circuit(4)
        .add_gate(OpType.CU1, 1.0, [0, 1])
        .add_gate(OpType.CU1, 1.0, [1, 2])
        .add_gate(OpType.CU1, 1.0, [2, 3])
    )

    small_network = NISQNetwork([[0, 1]], {0: [0, 1], 1: [2, 3, 4]})

    large_network = NISQNetwork(
        [[0, 1], [0, 2]], {0: [0, 1, 2], 1: [3, 4, 5], 2: [6, 7, 8, 9]}
    )

    allocator = Ordered()

    placement_one = Placement({0: 2, 1: 2, 2: 2})
    alloc_distribution_one = allocator.allocate(small_circ, large_network)
    placement_two = Placement({0: 1, 1: 1, 2: 1, 3: 0, 4: 1, 5: 1, 6: 1})
    alloc_distribution_two = allocator.allocate(med_circ, small_network)

    assert alloc_distribution_one.placement == placement_one
    assert alloc_distribution_two.placement == placement_two


def test_brute_distribute_small_hyperedge():

    network = NISQNetwork([[0, 1], [0, 2]], {0: [0], 1: [1, 2], 2: [3, 4]})

    circ = (
        Circuit(4)
        .add_gate(OpType.CU1, 1.0, [0, 3])
        .H(3)
        .add_gate(OpType.CU1, 1.0, [1, 3])
        .H(3)
        .add_gate(OpType.CU1, 1.0, [2, 3])
        .H(3)
    )

    allocator = Brute()
    distribution = allocator.allocate(circ, network)

    assert distribution.cost() == 3
    assert distribution.placement == Placement(
        {0: 0, 4: 0, 1: 1, 5: 0, 2: 2, 6: 2, 3: 2}
    )


def test_brute_distribute():

    small_network = NISQNetwork([[0, 1]], {0: [0, 1], 1: [2]})
    small_circ = Circuit(2).add_gate(OpType.CU1, 1.0, [0, 1])

    med_network = NISQNetwork([[0, 1], [0, 2]], {0: [0], 1: [1], 2: [2, 3]})
    med_circ = (
        Circuit(4)
        .add_gate(OpType.CU1, 1.0, [0, 1])
        .add_gate(OpType.CU1, 1.0, [1, 2])
        .add_gate(OpType.CU1, 1.0, [2, 3])
    )

    allocator = Brute()

    distribution_small = allocator.allocate(small_circ, small_network)
    assert distribution_small.placement == Placement({0: 0, 2: 0, 1: 0})
    assert distribution_small.cost() == 0

    distribution_med = allocator.allocate(med_circ, med_network)
    assert distribution_med.placement == Placement(
        {0: 0, 4: 0, 1: 1, 5: 0, 2: 2, 6: 2, 3: 2}
    )
    assert distribution_med.cost() == 2


def test_routing_allocator():

    small_network = NISQNetwork([[0, 1], [0, 2]], {0: [0], 1: [1], 2: [2, 3]})
    small_circ = (
        Circuit(4)
        .add_gate(OpType.CU1, 1.0, [0, 1])
        .add_gate(OpType.CU1, 1.0, [1, 2])
        .add_gate(OpType.CU1, 1.0, [2, 3])
    )

    allocator = Routing()
    routing_distribution = allocator.allocate(small_circ, small_network)
    ideal_placement = Placement({0: 0, 4: 1, 5: 0, 1: 1, 2: 2, 6: 2, 3: 2})
    assert routing_distribution.placement == ideal_placement
    assert routing_distribution.cost() == 2

    med_network = NISQNetwork([[0, 1]], {0: [0, 1], 1: [2, 3, 4]})
    med_circ = (
        Circuit(5)
        .add_gate(OpType.CU1, 1.0, [0, 1])
        .add_gate(OpType.CU1, 1.0, [1, 2])
        .add_gate(OpType.CU1, 1.0, [0, 2])
        .add_gate(OpType.CU1, 1.0, [2, 3])
        .add_gate(OpType.CU1, 1.0, [3, 4])
        .add_gate(OpType.CU1, 1.0, [3, 2])
    )
    routing_distribution = allocator.allocate(med_circ, med_network)
    ideal_placement = Placement(
        {0: 0, 8: 1, 9: 0, 10: 0, 1: 0, 2: 1, 6: 1, 7: 1, 3: 1, 5: 1, 4: 1}
    )

    assert routing_distribution.placement == ideal_placement
    assert routing_distribution.cost() == 2

    med_circ_flipped = (
        Circuit(5)
        .add_gate(OpType.CU1, 1.0, [0, 1])
        .add_gate(OpType.CU1, 1.0, [1, 2])
        .add_gate(OpType.CU1, 1.0, [0, 2])
        .add_gate(OpType.CU1, 1.0, [2, 3])
        .add_gate(OpType.CU1, 1.0, [3, 4])
        .add_gate(OpType.CU1, 1.0, [2, 3])
    )

    routing_distribution = allocator.allocate(
        med_circ_flipped, med_network
    )
    ideal_placement = Placement(
        {0: 0, 8: 1, 9: 0, 10: 1, 1: 0, 2: 1, 6: 1, 7: 1, 3: 1, 5: 1, 4: 1}
    )

    assert routing_distribution.placement == ideal_placement
    assert routing_distribution.cost() == 1


@pytest.mark.skip(reason="QControlBox are not supported for now")
def test_q_control_box_circuits():

    network = NISQNetwork([[0, 1]], {0: [0], 1: [1]})

    op = Op.create(OpType.V)
    cv = QControlBox(op, 1)

    circ = Circuit(2)
    circ.add_qcontrolbox(cv, [0, 1])
    circ.add_qcontrolbox(cv, [0, 1])
    circ.Rz(0.3, 0)
    circ.add_qcontrolbox(cv, [0, 1])
    circ.H(0)
    circ.add_qcontrolbox(cv, [0, 1])
    circ.add_qcontrolbox(cv, [0, 1])

    allocator = Brute()

    distribution = allocator.allocate(circ, network)
    ideal_placement = Placement({0: 0, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 1: 1})

    assert distribution.placement == ideal_placement
    assert distribution.cost() == 3


def test_CRz_circuits():

    network = NISQNetwork([[0, 1]], {0: [0], 1: [1]})

    circ = Circuit(2)
    circ.add_gate(OpType.CU1, 0.3, [1, 0])
    circ.add_gate(OpType.CU1, 0.1, [0, 1])
    circ.Rz(0.3, 0)
    circ.add_gate(OpType.CU1, 0.4, [0, 1])
    circ.H(0)
    circ.add_gate(OpType.CU1, 0.5, [1, 0])
    circ.add_gate(OpType.CU1, 0.2, [0, 1])

    allocator = Brute()

    distribution = allocator.allocate(circ, network)

    assert distribution.placement == Placement(
        {0: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 1: 1}
    )
    assert distribution.cost() == 1
