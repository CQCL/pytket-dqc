from pytket_dqc.networks import NISQNetwork
from pytket_dqc import DistributedCircuit
from pytket_dqc.placement import Placement
from pytket import Circuit


def test_placement_valid():

    large_network = NISQNetwork([[0, 1], [0, 2], [1, 2]], {
                                0: [0, 1, 2], 1: [3, 4, 5], 2: [6, 7, 8, 9]})
    small_network = NISQNetwork([[0, 1]], {0: [0, 1], 1: [2]})

    small_circ = Circuit(2).CZ(0, 1)
    dist_small_circ = DistributedCircuit(small_circ)

    med_circ = Circuit(4).CZ(0, 1).CZ(1, 2).CZ(2, 3)
    dist_med_circ = DistributedCircuit(med_circ)

    placement_one = Placement({0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1})
    placement_two = Placement({0: 1, 1: 1, 2: 1, 3: 0, 4: 0, 5: 0, 6: 2})
    placement_three = Placement({0: 2, 1: 2, 2: 2, 3: 2, 4: 0, 5: 0, 6: 0})
    placement_four = Placement({0: 0, 1: 0, 2: 0, 3: 2, 4: 0, 5: 0, 6: 0})
    placement_five = Placement({0: 0, 1: 0, 2: 0, 3: 0})
    placement_six = Placement({0: 2, 1: 2, 2: 3})
    placement_seven = Placement({0: 1, 1: 1, 2: 1, 3: 0, 4: 1, 5: 1, 6: 1})
    placement_eight = Placement({0: 1, 1: 0, 2: 1})
    placement_nine = Placement({0: 2, 1: 2, 2: 2})

    assert not placement_one.valid(dist_med_circ, large_network)
    assert placement_two.valid(dist_med_circ, large_network)
    assert placement_three.valid(dist_med_circ, large_network)
    assert placement_four.valid(dist_med_circ, large_network)
    assert not placement_five.valid(dist_med_circ, large_network)
    assert not placement_six.valid(dist_med_circ, large_network)
    assert not placement_seven.valid(dist_small_circ, small_network)
    assert placement_eight.valid(dist_small_circ, small_network)
    assert not placement_nine.valid(dist_small_circ, small_network)


def test_placement_cost():

    two_CZ_circ = Circuit(3).CZ(0, 1).CZ(0, 2)
    dist_two_CZ_circ = DistributedCircuit(two_CZ_circ)

    three_line_network = NISQNetwork(
        [[0, 1], [1, 2]], {0: [0], 1: [1], 2: [2]})

    placement_one = Placement({0: 0, 1: 1, 2: 2, 3: 1, 4: 2})
    assert placement_one.cost(
        dist_two_CZ_circ,
        three_line_network
    ) == 3
    placement_two = Placement({0: 0, 1: 1, 2: 2, 3: 1, 4: 0})
    assert placement_two.cost(
        dist_two_CZ_circ, three_line_network) == 3
    placement_three = Placement({0: 1, 1: 0, 2: 2, 3: 0, 4: 2})
    assert placement_three.cost(
        dist_two_CZ_circ, three_line_network) == 2
