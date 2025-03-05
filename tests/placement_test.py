from pytket_dqc.networks import NISQNetwork
from pytket_dqc import HypergraphCircuit
from pytket_dqc.placement import Placement
from pytket import Circuit, OpType


# TODO: Add tests with circuits where one or more qubits are unused


def test_placement_valid():
    large_network = NISQNetwork(
        [[0, 1], [0, 2], [1, 2]], {0: [0, 1, 2], 1: [3, 4, 5], 2: [6, 7, 8, 9]}
    )
    small_network = NISQNetwork([[0, 1]], {0: [0, 1], 1: [2]})

    small_circ = Circuit(2).add_gate(OpType.CU1, 1.0, [0, 1])
    dist_small_circ = HypergraphCircuit(small_circ)

    med_circ = (
        Circuit(4)
        .add_gate(OpType.CU1, 1.0, [0, 1])
        .add_gate(OpType.CU1, 1.0, [1, 2])
        .add_gate(OpType.CU1, 1.0, [2, 3])
    )
    dist_med_circ = HypergraphCircuit(med_circ)

    placement_one = Placement({0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1})
    placement_two = Placement({0: 1, 1: 1, 2: 1, 3: 0, 4: 0, 5: 0, 6: 2})
    placement_three = Placement({0: 2, 1: 2, 2: 2, 3: 2, 4: 0, 5: 0, 6: 0})
    placement_four = Placement({0: 0, 1: 0, 2: 0, 3: 2, 4: 0, 5: 0, 6: 0})
    placement_five = Placement({0: 0, 1: 0, 2: 0, 3: 0})
    placement_six = Placement({0: 2, 1: 2, 2: 3})
    placement_seven = Placement({0: 1, 1: 1, 2: 1, 3: 0, 4: 1, 5: 1, 6: 1})
    placement_eight = Placement({0: 1, 1: 0, 2: 1})
    placement_nine = Placement({0: 2, 1: 2, 2: 2})

    assert not placement_one.is_valid(dist_med_circ, large_network)
    assert placement_two.is_valid(dist_med_circ, large_network)
    assert placement_three.is_valid(dist_med_circ, large_network)
    assert placement_four.is_valid(dist_med_circ, large_network)
    assert not placement_five.is_valid(dist_med_circ, large_network)
    assert not placement_six.is_valid(dist_med_circ, large_network)
    assert not placement_seven.is_valid(dist_small_circ, small_network)
    assert placement_eight.is_valid(dist_small_circ, small_network)
    assert not placement_nine.is_valid(dist_small_circ, small_network)
