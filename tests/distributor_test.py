from pytket_dqc.distributors import Annealing
from pytket_dqc import DistributedCircuit
from pytket import Circuit
from pytket_dqc.networks import NISQNetwork
from pytket_dqc.distributors.annealing import order_reducing_size
from pytket_dqc.distributors import Brute
from pytket_dqc.distributors import Routing
from pytket_dqc.placement import Placement


def test_order_reducing_size():
    my_dict = {0: [0, 1], 2: [5, 6, 7, 8], 1: [2, 3, 4]}
    assert order_reducing_size(my_dict) == {
        2: [5, 6, 7, 8], 1: [2, 3, 4], 0: [0, 1]}


def test_annealing_initial_placement():

    small_circ = Circuit(2).CZ(0, 1)
    dist_small_circ = DistributedCircuit(small_circ)

    med_circ = Circuit(4).CZ(0, 1).CZ(1, 2).CZ(2, 3)
    dist_med_circ = DistributedCircuit(med_circ)

    small_network = NISQNetwork([[0, 1]], {0: [0, 1], 1: [2, 3, 4]})

    large_network = NISQNetwork(
        [[0, 1], [0, 2]], {0: [0, 1, 2], 1: [3, 4, 5], 2: [6, 7, 8, 9]})

    distributor = Annealing()

    placement_one = Placement({0: 2, 1: 2, 2: 2})
    distributor_placement_one = distributor.initial_placement(
        dist_small_circ, large_network)
    placement_two = Placement({0: 1, 1: 1, 2: 1, 3: 0, 4: 1, 5: 1, 6: 1})
    distributor_placement_two = distributor.initial_placement(
        dist_med_circ,
        small_network
    )

    assert distributor_placement_one == placement_one
    assert distributor_placement_two == placement_two


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
        {0: 2, 4: 2, 1: 2, 5: 2, 2: 0, 6: 1, 3: 1})
    assert placement_med.cost(dist_med_circ, med_network) == 2

# TODO: Add test of second circuit and network here


def test_routing_distribute():

    med_network = NISQNetwork([[0, 1], [0, 2]], {0: [0], 1: [1], 2: [2, 3]})
    med_circ = Circuit(4).CZ(0, 1).CZ(1, 2).CZ(2, 3)
    dist_med_circ = DistributedCircuit(med_circ)

    distributor = Routing()
    routing_placement = distributor.distribute(dist_med_circ, med_network)
    ideal_placement = Placement({0: 0, 4: 1, 5: 0, 1: 1, 2: 2, 6: 2, 3: 2})
    cost = routing_placement.cost(dist_med_circ, med_network)
    assert routing_placement == ideal_placement
    assert cost == 2
