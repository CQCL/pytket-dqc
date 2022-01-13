from pytket_dqc.networks import NISQNetwork, ServerNetwork
from pytket import Circuit
from pytket_dqc import DistributedCircuit


def test_nisq_get_nx():
    network = NISQNetwork(
        [[0, 1], [0, 2]], {0: [0, 1, 2], 1: [3, 4, 5], 2: [6, 7, 8, 9]})

    G_full = network.get_nisq_nx()
    G_server = network.get_server_nx()

    assert list(G_full.edges()) == [(0, 1), (0, 2), (0, 3), (0, 6), (1, 2), (
        3, 4), (3, 5), (4, 5), (6, 7), (6, 8), (6, 9), (7, 8), (7, 9), (8, 9)]
    assert list(G_server.edges()) == [(0, 1), (0, 2)]


def test_server_get_nx():

    server_network = ServerNetwork([[0, 1], [0, 2], [1, 2]])
    G = server_network.get_server_nx()
    assert list(G.edges()) == [(0, 1), (0, 2), (1, 2)]


def test_get_server_list():

    large_network = NISQNetwork([[0, 1], [0, 2], [1, 2]], {
                                0: [0, 1, 2], 1: [3, 4, 5], 2: [6, 7, 8, 9]})
    small_network = NISQNetwork([[0, 1]], {0: [0, 1], 1: [2, 3, 4]})

    assert small_network.get_server_list() == [0, 1]
    assert large_network.get_server_list() == [0, 1, 2]


def test_server_network_is_placement():

    large_network = ServerNetwork([[0, 1], [0, 2], [1, 2]])
    small_network = ServerNetwork([[0, 1]])

    assert large_network.is_placement(
        {0: 1, 1: 1, 2: 1, 3: 0, 4: 1, 5: 1, 6: 1})
    assert large_network.is_placement({0: 2, 1: 2, 2: 2})
    assert large_network.is_placement({0: 0, 1: 0, 2: 0})
    assert large_network.is_placement({0: 0, 1: 0, 2: 0, 3: 0})
    assert not large_network.is_placement({0: 2, 1: 2, 2: 3})
    assert small_network.is_placement(
        {0: 1, 1: 1, 2: 1, 3: 0, 4: 1, 5: 1, 6: 1})
    assert not small_network.is_placement({0: 2, 1: 2, 2: 2})


def test_nisq_network_is_placement():

    large_network = NISQNetwork([[0, 1], [0, 2], [1, 2]], {
                                0: [0, 1, 2], 1: [3, 4, 5], 2: [6, 7, 8, 9]})
    small_network = NISQNetwork([[0, 1]], {0: [0, 1], 1: [2]})

    small_circ = Circuit(2).CZ(0, 1)
    dist_small_circ = DistributedCircuit(small_circ)

    med_circ = Circuit(4).CZ(0, 1).CZ(1, 2).CZ(2, 3)
    dist_med_circ = DistributedCircuit(med_circ)

    assert not large_network.is_circuit_placement(
        {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1}, dist_med_circ)
    assert large_network.is_circuit_placement(
        {0: 1, 1: 1, 2: 1, 3: 0, 4: 0, 5: 0, 6: 2}, dist_med_circ)
    assert large_network.is_circuit_placement(
        {0: 2, 1: 2, 2: 2, 3: 2, 4: 0, 5: 0, 6: 0}, dist_med_circ)
    assert large_network.is_circuit_placement(
        {0: 0, 1: 0, 2: 0, 3: 2, 4: 0, 5: 0, 6: 0}, dist_med_circ)
    assert not large_network.is_circuit_placement(
        {0: 0, 1: 0, 2: 0, 3: 0}, dist_med_circ)
    assert not large_network.is_circuit_placement(
        {0: 2, 1: 2, 2: 3}, dist_med_circ)
    assert not small_network.is_circuit_placement(
        {0: 1, 1: 1, 2: 1, 3: 0, 4: 1, 5: 1, 6: 1}, dist_small_circ)
    assert small_network.is_circuit_placement(
        {0: 1, 1: 0, 2: 1}, dist_small_circ)
    assert not small_network.is_circuit_placement(
        {0: 2, 1: 2, 2: 2}, dist_small_circ)
