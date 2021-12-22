from pytket_dqc import DistributedCircuit, Hypergraph
from pytket import Circuit
from pytket_dqc.networks import NISQNetwork


def test_distributed_circuit():

    circ = Circuit(2).CZ(0, 1)
    dist_circ = DistributedCircuit(circ)

    assert dist_circ.get_circuit() == circ

    vertex_circuit_map = dist_circ.get_vertex_circuit_map()
    assert vertex_circuit_map[0] == 'qubit'
    assert vertex_circuit_map[1] == 'qubit'
    assert vertex_circuit_map[2] == 'gate'


def test_hypergraph():

    hypgra = Hypergraph()
    hypgra.add_vertex(0)
    hypgra.add_vertex(1)
    hypgra.add_vertex(2)
    hypgra.add_hyperedge([0, 1])
    hypgra.add_hyperedge([2, 1])
    assert hypgra.get_vertex_list() == [0, 1, 2]
    assert hypgra.get_hyperedge_list() == [[0, 1], [2, 1]]


def test_placement_cost():

    two_CZ_circ = Circuit(3).CZ(0, 1).CZ(0, 2)
    dist_two_CZ_circ = DistributedCircuit(two_CZ_circ)

    three_line_network = NISQNetwork(
        [[0, 1], [1, 2]], {0: [0], 1: [1], 2: [2]})

    assert dist_two_CZ_circ.placement_cost(
        {0: 0, 1: 1, 2: 2, 3: 1, 4: 2},
        three_line_network
    ) == 3


def test_hypergraph_is_placement():

    large_network = NISQNetwork([[0, 1], [0, 2], [1, 2]], {
                                0: [0, 1, 2], 1: [3, 4, 5], 2: [6, 7, 8, 9]})
    small_network = NISQNetwork([[0, 1]], {0: [0, 1], 1: [2, 3, 4]})

    small_circ = Circuit(2).CZ(0, 1)
    dist_small_circ = DistributedCircuit(small_circ)

    med_circ = Circuit(4).CZ(0, 1).CZ(1, 2).CZ(2, 3)
    dist_med_circ = DistributedCircuit(med_circ)

    assert dist_med_circ.is_placement(
        {0: 1, 1: 1, 2: 1, 3: 0, 4: 1, 5: 1, 6: 1}, large_network)
    assert dist_med_circ.is_placement(
        {0: 1, 1: 1, 2: 1, 3: 0, 4: 1, 5: 1, 6: 1}, small_network)
    assert not dist_med_circ.is_placement({0: 2, 1: 2, 2: 2}, large_network)
    assert not dist_med_circ.is_placement({0: 2, 1: 2, 2: 2}, small_network)
    assert not dist_small_circ.is_placement(
        {0: 1, 1: 1, 2: 1, 3: 0, 4: 1, 5: 1, 6: 1}, large_network)
    assert not dist_small_circ.is_placement(
        {0: 1, 1: 1, 2: 1, 3: 0, 4: 1, 5: 1, 6: 1}, small_network)
    assert dist_small_circ.is_placement({0: 2, 1: 2, 2: 2}, large_network)
    assert not dist_small_circ.is_placement({0: 2, 1: 2, 2: 2}, small_network)
