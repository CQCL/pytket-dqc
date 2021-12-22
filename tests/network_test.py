from pytket_dqc.networks import NISQNetwork, ServerNetwork


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
