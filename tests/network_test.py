from pytket_dqc.networks import NISQNetwork, ServerNetwork
from pytket.routing import Architecture, NoiseAwarePlacement  # type:ignore
from pytket.circuit import Node  # type:ignore
import pytest
from pytket_dqc.placement import Placement


def test_nisq_get_architecture():

    med_network = NISQNetwork([[0, 1], [0, 2]], {0: [0], 1: [1], 2: [2, 3]})
    med_arch = Architecture(
        [(Node(2), Node(3)), (Node(2), Node(0)), (Node(0), Node(1))])
    arc, node_qubit_map = med_network.get_architecture()
    assert arc == med_arch
    assert node_qubit_map == {Node(0): 0, Node(1): 1, Node(2): 2, Node(3): 3}


@pytest.mark.skip(reason="This test needs to be fixed.")
def test_nisq_get_placer():

    med_network = NISQNetwork([[0, 1], [0, 2]], {0: [0], 1: [1], 2: [2, 3]})
    med_arch = Architecture(
        [(Node(2), Node(3)), (Node(2), Node(0)), (Node(0), Node(1))])
    link_errors = {
        (Node(2), Node(3)): 0,
        (Node(2), Node(0)): 1,
        (Node(0), Node(1)): 1,
    }
    placer = NoiseAwarePlacement(arc=med_arch, link_errors=link_errors)
    assert med_network.get_placer() == placer


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

    placement_one = Placement({0: 1, 1: 1, 2: 1, 3: 0, 4: 1, 5: 1, 6: 1})
    placement_two = Placement({0: 2, 1: 2, 2: 2})
    placement_three = Placement({0: 0, 1: 0, 2: 0})
    placement_four = Placement({0: 0, 1: 0, 2: 0, 3: 0})
    placement_five = Placement({0: 2, 1: 2, 2: 3})

    assert large_network.is_placement(placement_one)
    assert large_network.is_placement(placement_two)
    assert large_network.is_placement(placement_three)
    assert large_network.is_placement(placement_four)
    assert not large_network.is_placement(placement_five)
    assert small_network.is_placement(placement_one)
    assert not small_network.is_placement(placement_two)
