from pytket_dqc.networks import NISQNetwork
import networkx as nx  # type: ignore


def test_get_nx():
    network = NISQNetwork(
        [[0, 1], [0, 2]], {0: [0, 1, 2], 1: [3, 4, 5], 2: [6, 7, 8, 9]})
    assert isinstance(network.get_nx(), nx.Graph)
