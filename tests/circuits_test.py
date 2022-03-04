from pytket_dqc import DistributedCircuit, Hypergraph
from pytket_dqc.circuits import RegularGraphDistributedCircuit
from pytket import Circuit
from pytket_dqc.placement import Placement
from pytket_dqc.distributors import Brute
from pytket_dqc.networks import NISQNetwork


# TODO: Test new circuit classes

def test_hypergraph_is_valid():

    hypgraph = Hypergraph()
    hypgraph.add_vertices([1, 2, 3])
    assert not hypgraph.is_valid()
    hypgraph.add_vertex(0)
    assert hypgraph.is_valid()


# TODO: Include vertex type information in this test
def test_distributed_circuit():

    circ = Circuit(2).CZ(0, 1)
    dist_circ = DistributedCircuit(circ)

    assert dist_circ.circuit == circ

    vertex_circuit_map = dist_circ.vertex_circuit_map
    assert vertex_circuit_map[0]['type'] == 'qubit'
    assert vertex_circuit_map[1]['type'] == 'qubit'
    assert vertex_circuit_map[2]['type'] == 'gate'


def test_regular_graph_distributed_circuit():

    dist_circ = RegularGraphDistributedCircuit(3, 2, 1, seed=0)
    network = NISQNetwork([[0, 1], [0, 2]], {0: [0, 1], 1: [2, 3, 4], 2: [5]})
    distributor = Brute()
    placement = distributor.distribute(dist_circ, network)
    cost = placement.cost(dist_circ, network)

    assert cost == 0
    assert placement == Placement({0: 1, 3: 1, 4: 1, 1: 1, 5: 1, 2: 1})


def test_hypergraph():

    hypgra = Hypergraph()
    hypgra.add_vertex(0)
    hypgra.add_vertex(1)
    hypgra.add_vertex(2)
    hypgra.add_hyperedge([0, 1])
    hypgra.add_hyperedge([2, 1])
    assert hypgra.vertex_list == [0, 1, 2]
    assert hypgra.hyperedge_list == [
        {"hyperedge": [0, 1], "weight":1}, {"hyperedge": [2, 1], "weight":1}]


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


def test_hypergrpah_kahypar_hyperedges():

    hypgraph = Hypergraph()

    hypgraph.add_vertices([i+1 for i in range(6)])
    hypgraph.add_hyperedge([3, 6, 2])
    hypgraph.add_hyperedge([3, 1])
    hypgraph.add_hyperedge([4, 5, 6])

    hyperedge_indices, hyperedges = hypgraph.kahypar_hyperedges()

    assert hyperedge_indices == [0, 3, 5, 8]
    assert hyperedges == [3, 6, 2, 3, 1, 4, 5, 6]
