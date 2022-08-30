from pytket_dqc.allocators import (
    Annealing,
    Random,
    Ordered,
    Routing,
    GraphPartitioning,
    Brute,
)
from pytket_dqc.allocators.annealing import acceptance_criterion
from pytket_dqc import HypergraphCircuit
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
        Circuit(4).CZ(0, 3).Rz(0.5, 3).CZ(1, 3).Rz(0.5, 3).CZ(2, 3).Rz(0.5, 3)
    )
    dist_circ = HypergraphCircuit(circ)

    alloc = Annealing()

    placement = alloc.allocate(
        dist_circ,
        network,
        seed=2,
        iterations=1,
        initial_place_method=Ordered(),
    )

    assert placement == Placement({0: 1, 1: 1, 2: 2, 3: 0, 4: 1, 5: 1, 6: 1})
    assert placement.cost(dist_circ, network) == 5

    placement = alloc.allocate(
        dist_circ,
        network,
        seed=1,
        iterations=1,
        initial_place_method=Ordered(),
    )

    assert placement == Placement({0: 1, 1: 1, 2: 2, 3: 2, 4: 0, 5: 1, 6: 1})
    assert placement.cost(dist_circ, network) == 8


def test_acceptance_criterion():

    assert acceptance_criterion(1, 10) >= 1
    assert acceptance_criterion(-1, 10) < 1


def test_graph_initial_partitioning():

    network = NISQNetwork([[0, 1], [0, 2]], {0: [0], 1: [1, 2], 2: [3, 4]})

    circ = (
        Circuit(4).CZ(0, 3).Rz(0.5, 3).CZ(1, 3).Rz(0.5, 3).CZ(2, 3).Rz(0.5, 3)
    )
    dist_circ = HypergraphCircuit(circ)

    alloc = GraphPartitioning()

    # num_rounds = 0 so that there are no refinement rounds
    initial_placement = alloc.allocate(
        dist_circ, network, seed=1, num_rounds=0
    )

    assert initial_placement == Placement(
        {0: 2, 1: 0, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1}
    ) or initial_placement == Placement(
        {0: 2, 1: 1, 2: 2, 3: 1, 4: 1, 5: 1, 6: 1}
    )


def test_graph_partitioning_refinement():

    network = NISQNetwork([[0, 1], [0, 2]], {0: [0], 1: [1, 2], 2: [3, 4, 5]})

    circ = (
        Circuit(4)
        .CZ(0, 3)
        .Rx(0.3, 0)
        .CZ(1, 3)
        .Rx(0.3, 1)
        .CZ(2, 3)
        .Rx(0.3, 2)
        .CZ(0, 3)
        .CZ(1, 3)
        .CZ(2, 3)
    )
    dist_circ = HypergraphCircuit(circ)

    alloc = GraphPartitioning()
    bad_placement = Placement({v: 2 for v in dist_circ.vertex_list})
    assert not bad_placement.is_valid(dist_circ, network)

    refined_placement = alloc.refine(
        bad_placement, dist_circ, network, seed=1
    )
    good_placement = Placement(
        {0: 2, 1: 2, 2: 2, 3: 0, 4: 2, 5: 2, 6: 2, 7: 2, 8: 2, 9: 2}
    )

    assert refined_placement.is_valid(dist_circ, network)
    assert refined_placement == good_placement


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

    dist_circ = HypergraphCircuit(circuit)
    alloc = GraphPartitioning()

    placement = alloc.allocate(dist_circ, network, seed=0)
    assert placement.is_valid(dist_circ, network)


def test_graph_partitioning_unused_qubits():

    network = NISQNetwork([[0, 1], [0, 2]], {0: [0], 1: [1, 2], 2: [3, 4]})
    alloc = GraphPartitioning()

    circ = Circuit(2)
    dist_circ = HypergraphCircuit(circ)

    placement = alloc.allocate(dist_circ, network, seed=1)
    assert placement == Placement({0: 1, 1: 0})

    circ = Circuit(0)
    dist_circ = HypergraphCircuit(circ)

    placement = alloc.allocate(dist_circ, network, seed=1)
    assert placement == Placement(dict())

    circ = Circuit(3).CZ(1, 2)
    dist_circ = HypergraphCircuit(circ)

    placement = alloc.allocate(dist_circ, network, seed=1)
    assert (
        (placement == Placement({0: 2, 1: 1, 2: 1, 3: 1})) or
        (placement == Placement({0: 1, 1: 2, 2: 2, 3: 2}))
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

    circ = Circuit(3).CZ(0, 2).CZ(1, 2)
    dist_circ = HypergraphCircuit(circ)

    network = NISQNetwork([[0, 1], [0, 2]], {0: [0, 1], 1: [2, 3], 2: [4]})

    alloc = Random()
    placement = alloc.allocate(dist_circ, network, seed=0)

    assert placement == Placement({0: 1, 3: 1, 1: 0, 4: 1, 2: 2})
    assert placement.cost(dist_circ, network) == 3


def test_ordered_allocator():

    small_circ = Circuit(2).CZ(0, 1)
    dist_small_circ = HypergraphCircuit(small_circ)

    med_circ = Circuit(4).CZ(0, 1).CZ(1, 2).CZ(2, 3)
    dist_med_circ = HypergraphCircuit(med_circ)

    small_network = NISQNetwork([[0, 1]], {0: [0, 1], 1: [2, 3, 4]})

    large_network = NISQNetwork(
        [[0, 1], [0, 2]], {0: [0, 1, 2], 1: [3, 4, 5], 2: [6, 7, 8, 9]}
    )

    alloc = Ordered()

    placement_one = Placement({0: 2, 1: 2, 2: 2})
    alloc_placement_one = alloc.allocate(
        dist_small_circ, large_network
    )
    placement_two = Placement({0: 1, 1: 1, 2: 1, 3: 0, 4: 1, 5: 1, 6: 1})
    alloc_placement_two = alloc.allocate(
        dist_med_circ, small_network
    )

    assert alloc_placement_one == placement_one
    assert alloc_placement_two == placement_two


def test_brute_distribute_small_hyperedge():

    network = NISQNetwork([[0, 1], [0, 2]], {0: [0], 1: [1, 2], 2: [3, 4]})

    circ = (
        Circuit(4).CZ(0, 3).Rz(0.5, 3).CZ(1, 3).Rz(0.5, 3).CZ(2, 3).Rz(0.5, 3)
    )
    dist_circ = HypergraphCircuit(circ)

    alloc = Brute()
    placement = alloc.allocate(dist_circ, network)

    assert placement.cost(dist_circ, network) == 3
    assert placement == Placement({0: 0, 4: 0, 1: 1, 5: 0, 2: 2, 6: 2, 3: 2})


def test_brute_distribute():

    small_network = NISQNetwork([[0, 1]], {0: [0, 1], 1: [2]})
    small_circ = Circuit(2).CZ(0, 1)
    dist_small_circ = HypergraphCircuit(small_circ)

    med_network = NISQNetwork([[0, 1], [0, 2]], {0: [0], 1: [1], 2: [2, 3]})
    med_circ = Circuit(4).CZ(0, 1).CZ(1, 2).CZ(2, 3)
    dist_med_circ = HypergraphCircuit(med_circ)

    alloc = Brute()

    placement_small = alloc.allocate(dist_small_circ, small_network)
    assert placement_small == Placement({0: 0, 2: 0, 1: 0})
    assert placement_small.cost(dist_small_circ, small_network) == 0

    placement_med = alloc.allocate(dist_med_circ, med_network)
    assert placement_med == Placement(
        {0: 0, 4: 0, 1: 1, 5: 0, 2: 2, 6: 2, 3: 2}
    )
    assert placement_med.cost(dist_med_circ, med_network) == 2


def test_routing_distribute():

    small_network = NISQNetwork([[0, 1], [0, 2]], {0: [0], 1: [1], 2: [2, 3]})
    small_circ = Circuit(4).CZ(0, 1).CZ(1, 2).CZ(2, 3)
    dist_small_circ = HypergraphCircuit(small_circ)

    alloc = Routing()
    routing_placement = alloc.allocate(dist_small_circ, small_network)
    ideal_placement = Placement({0: 0, 4: 1, 5: 0, 1: 1, 2: 2, 6: 2, 3: 2})
    cost = routing_placement.cost(dist_small_circ, small_network)
    assert routing_placement == ideal_placement
    assert cost == 2

    med_network = NISQNetwork([[0, 1]], {0: [0, 1], 1: [2, 3, 4]})
    med_circ = Circuit(5).CZ(0, 1).CZ(1, 2).CZ(0, 2).CZ(2, 3).CZ(3, 4).CZ(3, 2)
    dist_med_circ = HypergraphCircuit(med_circ)

    routing_placement = alloc.allocate(dist_med_circ, med_network)
    cost = routing_placement.cost(dist_med_circ, med_network)
    ideal_placement = Placement(
        {0: 0, 8: 1, 9: 0, 10: 0, 1: 0, 2: 1, 6: 1, 7: 1, 3: 1, 5: 1, 4: 1}
    )

    assert routing_placement == ideal_placement
    assert cost == 2

    med_circ_flipped = (
        Circuit(5).CZ(0, 1).CZ(1, 2).CZ(0, 2).CZ(2, 3).CZ(3, 4).CZ(2, 3)
    )
    dist_med_circ_flipped = HypergraphCircuit(med_circ_flipped)

    routing_placement = alloc.allocate(
        dist_med_circ_flipped, med_network
    )
    cost = routing_placement.cost(dist_med_circ_flipped, med_network)
    ideal_placement = Placement(
        {0: 0, 8: 1, 9: 0, 10: 1, 1: 0, 2: 1, 6: 1, 7: 1, 3: 1, 5: 1, 4: 1}
    )

    assert routing_placement == ideal_placement
    assert cost == 1


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
    circ.Rx(0.3, 0)
    circ.add_qcontrolbox(cv, [0, 1])
    circ.add_qcontrolbox(cv, [0, 1])

    dist_circ = HypergraphCircuit(circ)

    alloc = Brute()

    placement = alloc.allocate(dist_circ, network)
    ideal_placement = Placement({0: 0, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 1: 1})

    assert placement == ideal_placement
    assert placement.cost(dist_circ, network) == 3


@pytest.mark.skip(reason="Support for CRz gates temporarily disabled")
def test_CRz_circuits():

    network = NISQNetwork([[0, 1]], {0: [0], 1: [1]})

    circ = Circuit(2)
    circ.CRz(0.3, 1, 0)
    circ.CRz(0.1, 0, 1)
    circ.Rz(0.3, 0)
    circ.CRz(0.4, 0, 1)
    circ.Rx(0.3, 0)
    circ.CRz(0.5, 1, 0)
    circ.CRz(0.2, 0, 1)

    dist_circ = HypergraphCircuit(circ)

    alloc = Brute()

    placement = alloc.allocate(dist_circ, network)

    assert placement == Placement({0: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 1: 1})
    assert placement.cost(dist_circ, network) == 1
