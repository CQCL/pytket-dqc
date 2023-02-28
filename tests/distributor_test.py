from pytket_dqc.distributors import (
    PartitioningHeterogeneousEmbedding,
    CoverEmbeddingSteinerDetached,
    PartitioningAnnealing,
)
from pytket import Circuit
from pytket_dqc import NISQNetwork
from pytket_dqc.utils import DQCPass


def small_circuit_network():

    circ = Circuit(3).CZ(0, 1).CZ(1, 2)
    DQCPass().apply(circ)
    network = NISQNetwork(
        server_coupling=[[0, 1], [1, 2]],
        server_qubits={0: [0], 1: [1], 2: [2]},
    )
    return circ, network


def test_vertex_cover_steiner():

    circ, network = small_circuit_network()
    dist = CoverEmbeddingSteinerDetached().distribute(
        circ=circ,
        network=network,
        seed=0,
    )
    assert dist.is_valid()


def test_partitioning_embedding():

    circ, network = small_circuit_network()
    dist = PartitioningHeterogeneousEmbedding().distribute(
        circ=circ,
        network=network,
        n_rounds=5,
    )
    assert dist.is_valid()

    dist = PartitioningHeterogeneousEmbedding().distribute(
        circ=circ,
        network=network,
        n_rounds=5,
        initial_distributor=PartitioningAnnealing(),
    )
    assert dist.is_valid()
