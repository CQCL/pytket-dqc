from pytket_dqc import DistributedCircuit, Hypergraph
from pytket import Circuit
from pytket_dqc.placement import Placement


# TODO: Include vertex type information in this test
def test_distributed_circuit():

    circ = Circuit(2).CZ(0, 1)
    dist_circ = DistributedCircuit(circ)

    assert dist_circ.circuit == circ

    vertex_circuit_map = dist_circ.vertex_circuit_map
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
    assert hypgra.vertex_list == [0, 1, 2]
    assert hypgra.hyperedge_list == [[0, 1], [2, 1]]


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
