from pytket_dqc import DistributedCircuit, Hypergraph
from pytket import Circuit


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
