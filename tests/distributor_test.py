from pytket_dqc.distributors import Annealing
from pytket_dqc import DistributedCircuit
from pytket import Circuit
from pytket_dqc.networks import NISQNetwork
from pytket_dqc.distributors.annealing import order_reducing_size


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

    assert distributor.initial_placement(dist_small_circ, large_network) == {
        0: 2, 1: 2, 2: 2}
    assert distributor.initial_placement(dist_med_circ, small_network) == {
        0: 1, 1: 1, 2: 1, 3: 0, 4: 1, 5: 1, 6: 1}
