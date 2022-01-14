from pytket_dqc import DistributedCircuit, Hypergraph
from pytket import Circuit
from pytket_dqc.networks import NISQNetwork
from pytket_dqc.placement import Placement


# TODO: Include vertex type information in this test
def test_distributed_circuit():

    circ = Circuit(2).CZ(0, 1)
    dist_circ = DistributedCircuit(circ)

    assert dist_circ.get_circuit() == circ

    vertex_circuit_map = dist_circ.get_vertex_circuit_map()
    assert vertex_circuit_map[0]['type'] == 'qubit'
    assert vertex_circuit_map[1]['type'] == 'qubit'
    assert vertex_circuit_map[2]['type'] == 'gate'


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

    placement_one = Placement({0: 0, 1: 1, 2: 2, 3: 1, 4: 2})
    assert dist_two_CZ_circ.placement_cost(
        placement_one,
        three_line_network
    ) == 3
    placement_two = Placement({0: 0, 1: 1, 2: 2, 3: 1, 4: 0})
    assert dist_two_CZ_circ.placement_cost(
        placement_two, three_line_network) == 3
    placement_three = Placement({0: 1, 1: 0, 2: 2, 3: 0, 4: 2})
    assert dist_two_CZ_circ.placement_cost(
        placement_three, three_line_network) == 2


def test_hypergraph_is_placement():

    small_circ = Circuit(2).CZ(0, 1)
    dist_small_circ = DistributedCircuit(small_circ)

    med_circ = Circuit(4).CZ(0, 1).CZ(1, 2).CZ(2, 3)
    dist_med_circ = DistributedCircuit(med_circ)

    placement_one = Placement({0: 1, 1: 1, 2: 1, 3: 0, 4: 1, 5: 1, 6: 1})
    assert dist_med_circ.is_placement(placement_one)
    assert not dist_small_circ.is_placement(placement_one)

    placement_two = Placement({0: 2, 1: 2, 2: 2})
    assert not dist_med_circ.is_placement(placement_two)
    assert dist_small_circ.is_placement(placement_two)
