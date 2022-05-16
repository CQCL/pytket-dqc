from pytket_dqc.distributors import (
    Annealing,
    Random,
    Ordered,
    Routing,
    GraphPartitioning,
    Brute
)
from pytket_dqc.distributors.annealing import acceptance_criterion
from pytket_dqc import DistributedCircuit
from pytket import Circuit
from pytket_dqc.networks import NISQNetwork, ServerNetwork
from pytket_dqc.distributors.ordered import order_reducing_size
from pytket_dqc.placement import Placement
import kahypar as kahypar  # type:ignore
from pytket.circuit import QControlBox, Op, OpType  # type:ignore


# TODO: Test that the placement returned by routing does not
# have any cost two edges

def test_annealing_distribute():

    network = NISQNetwork([[0, 1], [0, 2]], {0: [0], 1: [1, 2], 2: [3, 4]})

    circ = Circuit(4).CZ(0, 3).Rz(0.5, 3).CZ(
        1, 3).Rz(0.5, 3).CZ(2, 3).Rz(0.5, 3)
    dist_circ = DistributedCircuit(circ)

    distributor = Annealing()

    placement = distributor.distribute(
        dist_circ,
        network,
        seed=2,
        iterations=1,
        initial_place_method=Ordered()
    )

    assert placement == Placement({0: 1, 1: 1, 2: 2, 3: 0, 4: 1, 5: 1, 6: 1})
    assert placement.cost(dist_circ, network) == 5

    placement = distributor.distribute(
        dist_circ,
        network,
        seed=1,
        iterations=1,
        initial_place_method=Ordered()
    )

    assert placement == Placement({0: 1, 1: 1, 2: 2, 3: 2, 4: 0, 5: 1, 6: 1})
    assert placement.cost(dist_circ, network) == 8


def test_acceptance_criterion():

    assert acceptance_criterion(0, 1, 10) >= 1
    assert acceptance_criterion(1, 0, 10) < 1


def test_graph_partitioning():

    network = NISQNetwork([[0, 1], [0, 2]], {0: [0], 1: [1, 2], 2: [3, 4]})

    circ = Circuit(4).CZ(0, 3).Rz(0.5, 3).CZ(
        1, 3).Rz(0.5, 3).CZ(2, 3).Rz(0.5, 3)
    dist_circ = DistributedCircuit(circ)

    distributor = GraphPartitioning()

    placement = distributor.distribute(dist_circ, network)
    cost = placement.cost(dist_circ, network)
    optimal = 3
    print(cost)

    assert cost == optimal


def test_kahypar_install():

    num_nodes = 4
    num_nets = 3

    hyperedge_indices = [0, 2, 4, 6]
    hyperedges = [0, 1, 1, 2, 2, 3]

    k = 2

    hypergraph = kahypar.Hypergraph(
        num_nodes, num_nets, hyperedge_indices, hyperedges, k)

    context = kahypar.Context()
    context.loadINIconfiguration("km1_kKaHyPar_sea20.ini")

    context.setK(k)
    context.setEpsilon(0.03)
    context.suppressOutput(True)

    kahypar.partition(hypergraph, context)
    placement = [hypergraph.blockID(i) for i in range(hypergraph.numNodes())]

    assert (placement == [0, 0, 1, 1] or placement == [1, 1, 0, 0])


def test_order_reducing_size():
    my_dict = {0: [0, 1], 2: [5, 6, 7, 8], 1: [2, 3, 4]}
    assert order_reducing_size(my_dict) == {
        2: [5, 6, 7, 8], 1: [2, 3, 4], 0: [0, 1]}


def test_random_distributor():

    circ = Circuit(3).CZ(0, 2).CZ(1, 2)
    dist_circ = DistributedCircuit(circ)

    network = NISQNetwork([[0, 1], [0, 2]], {0: [0, 1], 1: [2, 3], 2: [4]})

    distributor = Random()
    placement = distributor.distribute(
        dist_circ, network, seed=0)

    assert placement == Placement({0: 1, 3: 1, 1: 0, 4: 1, 2: 2})
    assert placement.cost(dist_circ, network) == 3


def test_ordered_distributor():

    small_circ = Circuit(2).CZ(0, 1)
    dist_small_circ = DistributedCircuit(small_circ)

    med_circ = Circuit(4).CZ(0, 1).CZ(1, 2).CZ(2, 3)
    dist_med_circ = DistributedCircuit(med_circ)

    small_network = NISQNetwork([[0, 1]], {0: [0, 1], 1: [2, 3, 4]})

    large_network = NISQNetwork(
        [[0, 1], [0, 2]], {0: [0, 1, 2], 1: [3, 4, 5], 2: [6, 7, 8, 9]})

    distributor = Ordered()

    placement_one = Placement({0: 2, 1: 2, 2: 2})
    distributor_placement_one = distributor.distribute(
        dist_small_circ, large_network)
    placement_two = Placement({0: 1, 1: 1, 2: 1, 3: 0, 4: 1, 5: 1, 6: 1})
    distributor_placement_two = distributor.distribute(
        dist_med_circ,
        small_network
    )

    assert distributor_placement_one == placement_one
    assert distributor_placement_two == placement_two


def test_brute_distribute_small_hyperedge():

    network = NISQNetwork([[0, 1], [0, 2]], {0: [0], 1: [1, 2], 2: [3, 4]})

    circ = Circuit(4).CZ(0, 3).Rz(0.5, 3).CZ(
        1, 3).Rz(0.5, 3).CZ(2, 3).Rz(0.5, 3)
    dist_circ = DistributedCircuit(circ)

    distributor = Brute()
    placement = distributor.distribute(dist_circ, network)

    assert placement.cost(dist_circ, network) == 3
    assert placement == Placement({0: 0, 4: 0, 1: 1, 5: 0, 2: 2, 6: 2, 3: 2})


def test_brute_distribute():

    small_network = NISQNetwork([[0, 1]], {0: [0, 1], 1: [2]})
    small_circ = Circuit(2).CZ(0, 1)
    dist_small_circ = DistributedCircuit(small_circ)

    med_network = NISQNetwork([[0, 1], [0, 2]], {0: [0], 1: [1], 2: [2, 3]})
    med_circ = Circuit(4).CZ(0, 1).CZ(1, 2).CZ(2, 3)
    dist_med_circ = DistributedCircuit(med_circ)

    distributor = Brute()

    placement_small = distributor.distribute(dist_small_circ, small_network)
    assert placement_small == Placement({0: 0, 2: 0, 1: 0})
    assert placement_small.cost(dist_small_circ, small_network) == 0

    placement_med = distributor.distribute(dist_med_circ, med_network)
    assert placement_med == Placement(
        {0: 0, 4: 0, 1: 1, 5: 0, 2: 2, 6: 2, 3: 2})
    assert placement_med.cost(dist_med_circ, med_network) == 2


def test_routing_distribute():

    small_network = NISQNetwork([[0, 1], [0, 2]], {0: [0], 1: [1], 2: [2, 3]})
    small_circ = Circuit(4).CZ(0, 1).CZ(1, 2).CZ(2, 3)
    dist_small_circ = DistributedCircuit(small_circ)

    distributor = Routing()
    routing_placement = distributor.distribute(dist_small_circ, small_network)
    ideal_placement = Placement({0: 0, 4: 1, 5: 0, 1: 1, 2: 2, 6: 2, 3: 2})
    cost = routing_placement.cost(dist_small_circ, small_network)
    assert routing_placement == ideal_placement
    assert cost == 2

    med_network = NISQNetwork([[0, 1]], {0: [0, 1], 1: [2, 3, 4]})
    med_circ = Circuit(5).CZ(0, 1).CZ(1, 2).CZ(0, 2).CZ(2, 3).CZ(3, 4).CZ(3, 2)
    dist_med_circ = DistributedCircuit(med_circ)

    routing_placement = distributor.distribute(dist_med_circ, med_network)
    cost = routing_placement.cost(dist_med_circ, med_network)
    ideal_placement = Placement(
        {0: 0, 8: 1, 9: 0, 10: 0, 1: 0, 2: 1, 6: 1, 7: 1, 3: 1, 5: 1, 4: 1})

    assert routing_placement == ideal_placement
    assert cost == 2

    med_circ_flipped = Circuit(5).CZ(0, 1).CZ(
        1, 2).CZ(0, 2).CZ(2, 3).CZ(3, 4).CZ(2, 3)
    dist_med_circ_flipped = DistributedCircuit(med_circ_flipped)

    routing_placement = distributor.distribute(
        dist_med_circ_flipped, med_network)
    cost = routing_placement.cost(dist_med_circ_flipped, med_network)
    ideal_placement = Placement(
        {0: 0, 8: 1, 9: 0, 10: 1, 1: 0, 2: 1, 6: 1, 7: 1, 3: 1, 5: 1, 4: 1})

    assert routing_placement == ideal_placement
    assert cost == 1


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

    dist_circ = DistributedCircuit(circ)

    distributor = Brute()

    placement = distributor.distribute(dist_circ, network)
    ideal_placement = Placement({0: 0, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 1: 1})

    assert placement == ideal_placement
    assert placement.cost(dist_circ, network) == 3
