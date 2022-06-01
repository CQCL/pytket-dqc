from pytket_dqc.networks import NISQNetwork
from pytket_dqc.circuits import Hypergraph
from pytket_dqc.placement import Placement
from pytket_dqc.distributors import GainManager

simple_hypergraph = Hypergraph()
simple_hypergraph.add_vertices(list(range(1, 10)))
simple_hypergraph.add_hyperedge([1, 3, 4])
simple_hypergraph.add_hyperedge([1, 2, 4])
simple_hypergraph.add_hyperedge([3, 4, 8])
simple_hypergraph.add_hyperedge([4, 5, 7, 8])
simple_hypergraph.add_hyperedge([3, 7, 9])
simple_hypergraph.add_hyperedge([5, 6, 7])
simple_hypergraph.add_hyperedge([2, 5, 6, 9])

t_network = NISQNetwork(
    [[1, 2], [2, 4], [3, 4], [4, 5]],
    {1: [1, 2, 3, 4], 2: [5], 3: [6, 7, 8], 4: [9], 5: [10, 11, 12]},
)
house_network = NISQNetwork(
    [[1, 2], [2, 4], [1, 3], [3, 4], [3, 5], [4, 5]],
    {1: [1, 2], 2: [3, 4], 3: [5, 6], 4: [7, 8], 5: [9, 10, 11, 12]},
)

qubit_vertices = frozenset(simple_hypergraph.vertex_list)

def test_moves():
    placement = Placement({1: 1, 2: 1, 3: 2, 4: 3, 5: 3, 6: 4, 7: 5, 8: 5, 9: 5})
    t_manager = GainManager(simple_hypergraph, qubit_vertices, t_network, placement)

    assert t_manager.occupancy[3] == 2
    assert t_manager.is_move_valid(1,3)
    t_manager.move(1,3)
    # Server 3 is now full!
    assert t_manager.occupancy[3] == 3
    assert not t_manager.is_move_valid(9,3)
    # However, since qubit 1 is already in 3, the following is valid
    assert t_manager.is_move_valid(1,3)
    # Server 2 is full
    assert t_manager.occupancy[2] == 1
    assert not t_manager.is_move_valid(1,2)
    # Let's make some spare space in server 2
    t_manager.move(3,1)
    assert t_manager.occupancy[2] == 0
    # Now the move should be valid
    assert t_manager.is_move_valid(1,2)
    t_manager.move(1,2)
    assert t_manager.occupancy[2] == 1
    assert t_manager.occupancy[3] == 2

def test_gain():
    placement = Placement({1: 1, 2: 1, 3: 2, 4: 3, 5: 3, 6: 4, 7: 5, 8: 5, 9: 5})
    t_manager = GainManager(simple_hypergraph, qubit_vertices, t_network, placement)

    # Moving a vertex to where it already is has no gain
    assert t_manager.gain(1,1) == 0

    steiner_cost_1235 = t_manager.steiner_cost(frozenset([1,2,3,5]))
    steiner_cost_123 = t_manager.steiner_cost(frozenset([1,2,3]))
    steiner_cost_135 = t_manager.steiner_cost(frozenset([1,3,5]))
    steiner_cost_235 = t_manager.steiner_cost(frozenset([2,3,5]))
    steiner_cost_13 = t_manager.steiner_cost(frozenset([1,3]))
    steiner_cost_35 = t_manager.steiner_cost(frozenset([3,5]))
    assert steiner_cost_1235 == 4
    assert steiner_cost_123 == 3
    assert steiner_cost_135 == 4
    assert steiner_cost_235 == 3
    assert steiner_cost_13 == 3
    assert steiner_cost_35 == 2

    # Make some space in server 5
    assert t_manager.is_move_valid(8,1)
    t_manager.move(8,1)
    assert t_manager.is_move_valid(9,1)
    t_manager.move(9,1)

    # The two hyperedges incident to vertex 1 connect servers [1,2,3] and [1,3]
    # If I do the following move:
    assert t_manager.is_move_valid(1,5)
    # then the two hyperedges would connect servers [2,3,5] and [1,3,5]
    # the gains should look like this:
    current_cost = steiner_cost_123 + steiner_cost_13
    new_cost = steiner_cost_235 + steiner_cost_135
    assert t_manager.gain(1,5) == current_cost - new_cost
    # Instead, let's move vertex 3 to server 5 so that server 2 is empty
    assert t_manager.is_move_valid(3,5)
    t_manager.move(3,5)
    assert t_manager.occupancy[2] == 0
    # Now that the two hyperedges incident in 1 connect servers [1,3,5]


