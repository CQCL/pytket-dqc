from pytket_dqc.networks import NISQNetwork


def test_get_nx():
    network = NISQNetwork(
        [[0, 1], [0, 2]], {0: [0, 1, 2], 1: [3, 4, 5], 2: [6, 7, 8, 9]})

    G_full = network.get_full_nx()
    G_server = network.get_server_nx()

    assert list(G_full.edges()) == [(0, 1), (0, 2), (0, 3), (0, 6), (1, 2), (
        3, 4), (3, 5), (4, 5), (6, 7), (6, 8), (6, 9), (7, 8), (7, 9), (8, 9)]
    assert list(G_server.edges()) == [(0, 1), (0, 2)]
