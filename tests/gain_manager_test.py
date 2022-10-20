from pytket_dqc.networks import NISQNetwork
from pytket_dqc.circuits import (
    Hyperedge,
    HypergraphCircuit,
    Distribution,
)
from pytket_dqc.placement import Placement
from pytket_dqc.allocators import GainManager
from pytket_dqc.utils import steiner_tree
from pytket import Circuit, OpType
from copy import copy

circ = Circuit(4)
circ.add_gate(OpType.CU1, 0.1234, [1, 2])
circ.add_gate(OpType.CU1, 0.1234, [0, 2])
circ.add_gate(OpType.CU1, 0.1234, [2, 3])
circ.add_gate(OpType.CU1, 0.1234, [0, 3])
circ.H(0).H(2).Rz(0.1234, 3)
circ.add_gate(OpType.CU1, 1.0, [0, 2])
circ.add_gate(OpType.CU1, 1.0, [0, 3])
circ.add_gate(OpType.CU1, 1.0, [1, 2])
circ.H(0).H(2).Rz(0.1234, 0)
circ.add_gate(OpType.CU1, 0.1234, [0, 1])
circ.add_gate(OpType.CU1, 0.1234, [0, 3])
circ.add_gate(OpType.CU1, 1.0, [1, 2])

network = NISQNetwork(
    [[0, 1], [0, 2], [0, 3], [3, 4]],
    {0: [0], 1: [1, 2], 2: [3, 4], 3: [7], 4: [5, 6]},
)


def test_placement_move_occupancy():
    placement = Placement(
        {
            0: 1,
            1: 1,
            2: 2,
            3: 4,
            4: 1,
            5: 1,
            6: 1,
            7: 1,
            8: 2,
            9: 4,
            10: 0,
            11: 1,
            12: 3,
            13: 0,
        }
    )
    distribution = Distribution(HypergraphCircuit(circ), placement, network)
    manager = GainManager(distribution)

    assert manager.occupancy[4] == 1
    assert manager.is_move_valid(2, 4)
    manager.move(2, 4)
    # Server 4 is now full!
    assert manager.occupancy[4] == 2
    assert not manager.is_move_valid(1, 4)
    # However, since qubit 3 is already in 4, the following is valid
    assert manager.is_move_valid(3, 4)
    # Moreover, gate vertices can always be moved around
    assert manager.is_move_valid(8, 4)
    # Restore original placement
    manager.move(2, 2)

    # Server 1 is full
    assert manager.occupancy[1] == len(network.server_qubits[1])
    assert not manager.is_move_valid(3, 1)
    # Let's make some spare space in server 1
    manager.move(0, 0)
    assert manager.occupancy[1] < len(network.server_qubits[1])
    # Now the move should be valid
    assert manager.is_move_valid(3, 1)
    manager.move(3, 1)
    assert manager.occupancy[0] == 1
    assert manager.occupancy[1] == 2
    assert manager.occupancy[4] == 0


def test_gain_no_embeddings():
    placement = Placement(
        {
            0: 1,
            1: 1,
            2: 2,
            3: 4,
            4: 1,
            5: 1,
            6: 1,
            7: 1,
            8: 2,
            9: 4,
            10: 0,
            11: 1,
            12: 3,
            13: 0,
        }
    )
    distribution = Distribution(HypergraphCircuit(circ), placement, network)
    manager = GainManager(distribution)

    # Moving a vertex to where it already is has no gain
    assert manager.move_gain(0, 1) == 0

    g = manager.server_graph

    steiner_cost_13 = len(steiner_tree(g, [1, 3]).edges)
    steiner_cost_34 = len(steiner_tree(g, [3, 4]).edges)
    steiner_cost_14 = len(steiner_tree(g, [1, 4]).edges)
    steiner_cost_4 = len(steiner_tree(g, [4]).edges)
    steiner_cost_1 = len(steiner_tree(g, [1]).edges)
    steiner_cost_124 = len(steiner_tree(g, [1, 2, 4]).edges)
    steiner_cost_12 = len(steiner_tree(g, [1, 2]).edges)
    steiner_cost_123 = len(steiner_tree(g, [1, 2, 3]).edges)
    steiner_cost_24 = len(steiner_tree(g, [2, 4]).edges)
    assert steiner_cost_13 == 2
    assert steiner_cost_34 == 1
    assert steiner_cost_14 == 3
    assert steiner_cost_4 == 0
    assert steiner_cost_1 == 0
    assert steiner_cost_124 == 4
    assert steiner_cost_12 == 2
    assert steiner_cost_123 == 3
    assert steiner_cost_24 == 3

    # The two hyperedges incident to vertex 12 are [0,11,12] and [3,9,12]
    # which are allocated to servers as follows:   [1, 1, 3] and [4,4, 3].
    # If we do the following move:
    assert manager.is_move_valid(12, 4)
    # then the allocation is as follows:           [1, 1, 4] and [4,4, 4].
    # The gains should look like this:
    current_cost = steiner_cost_13 + steiner_cost_34
    new_cost = steiner_cost_14 + steiner_cost_4
    assert manager.move_gain(12, 4) == current_cost - new_cost
    # Check that, indeed, the costs are updated accordingly
    assert (
        manager.hyperedge_cost_map[Hyperedge([0, 11, 12])] == steiner_cost_13
    )
    assert manager.hyperedge_cost_map[Hyperedge([3, 9, 12])] == steiner_cost_34
    manager.move(12, 4)
    assert (
        manager.hyperedge_cost_map[Hyperedge([0, 11, 12])] == steiner_cost_14
    )
    assert manager.hyperedge_cost_map[Hyperedge([3, 9, 12])] == steiner_cost_4
    # Move back
    manager.move(12, 3)

    # The hyperedges incident to vertex 0 are [0,5,7], [0,11,12] and [0,8,9]
    # which are allocated to servers:         [1,1,1], [1, 1, 3] and [1,2,4].
    # If we do the following move:
    assert manager.is_move_valid(0, 2)
    # then the allocation is as folows:       [2,1,1], [2, 1, 3] and [2,2,4].
    # The gains should look like this:
    current_cost = steiner_cost_1 + steiner_cost_13 + steiner_cost_124
    new_cost = steiner_cost_12 + steiner_cost_123 + steiner_cost_24
    assert manager.move_gain(0, 2) == current_cost - new_cost
    # Check that, indeed, the costs are updated accordingly
    assert manager.hyperedge_cost_map[Hyperedge([0, 5, 7])] == steiner_cost_1
    assert (
        manager.hyperedge_cost_map[Hyperedge([0, 11, 12])] == steiner_cost_13
    )
    assert manager.hyperedge_cost_map[Hyperedge([0, 8, 9])] == steiner_cost_124
    manager.move(0, 2)
    assert manager.hyperedge_cost_map[Hyperedge([0, 5, 7])] == steiner_cost_12
    assert (
        manager.hyperedge_cost_map[Hyperedge([0, 11, 12])] == steiner_cost_123
    )
    assert manager.hyperedge_cost_map[Hyperedge([0, 8, 9])] == steiner_cost_24
    # Move back
    manager.move(0, 1)

    # Check that the cache is correct
    assert (
        manager.steiner_cache[frozenset([1, 3])].edges
        == steiner_tree(g, [1, 3]).edges
    )
    assert (
        manager.steiner_cache[frozenset([3, 4])].edges
        == steiner_tree(g, [3, 4]).edges
    )
    assert (
        manager.steiner_cache[frozenset([1, 4])].edges
        == steiner_tree(g, [1, 4]).edges
    )
    assert (
        manager.steiner_cache[frozenset([4])].edges
        == steiner_tree(g, [4]).edges
    )
    assert (
        manager.steiner_cache[frozenset([1])].edges
        == steiner_tree(g, [1]).edges
    )
    assert (
        manager.steiner_cache[frozenset([1, 2, 4])].edges
        == steiner_tree(g, [1, 2, 4]).edges
    )
    assert (
        manager.steiner_cache[frozenset([1, 2])].edges
        == steiner_tree(g, [1, 2]).edges
    )
    assert (
        manager.steiner_cache[frozenset([1, 2, 3])].edges
        == steiner_tree(g, [1, 2, 3]).edges
    )
    assert (
        manager.steiner_cache[frozenset([2, 4])].edges
        == steiner_tree(g, [2, 4]).edges
    )


def test_split_merge():

    circ = Circuit(3)
    circ.add_gate(OpType.CU1, 1.0, [0, 1])
    circ.add_gate(OpType.CU1, 1.0, [0, 2])
    hyp_circ = HypergraphCircuit(circ)

    merge_hyperedge_list = copy(hyp_circ.hyperedge_list)

    old_hyperedge = Hyperedge(vertices=[0, 3, 4], weight=1)
    to_merge_hyperedge_one = Hyperedge(vertices=[0, 3], weight=1)
    to_merge_hyperedge_two = Hyperedge(vertices=[0, 4], weight=1)

    placement = Placement({0: 1, 1: 2, 2: 3, 3: 2, 4: 3})
    network = NISQNetwork(
        server_coupling=[[0, 1], [0, 2], [0, 3]],
        server_qubits={0: [0], 1: [1], 2: [2], 3: [3]}
    )
    distribution = Distribution(hyp_circ, placement, network)
    gain_manager = GainManager(distribution)
    gain_manager.split_hyperedge(
        old_hyperedge=old_hyperedge,
        new_hyperedge_list=[to_merge_hyperedge_one, to_merge_hyperedge_two]
    )
    split_hyperedge_list = gain_manager.distribution.circuit.hyperedge_list

    merge_gain = gain_manager.merge_hyperedge_gain(
        to_merge_hyperedge_list=[
            to_merge_hyperedge_one,
            to_merge_hyperedge_two
        ]
    )
    assert merge_gain == 1
    # Check that the hypergraph is not changed by this gain calculation
    assert (
        gain_manager.distribution.circuit.hyperedge_list ==
        split_hyperedge_list
    )

    gain_manager.merge_hyperedge(
        to_merge_hyperedge_list=[
            to_merge_hyperedge_one,
            to_merge_hyperedge_two
        ]
    )

    # Check that merge recovers original circuit. I assume the order of
    # hyperedges in the hyperedge_list does not matter. This may be reckless.
    merge_hyperedge_list.sort()
    gain_manager.distribution.circuit.hyperedge_list.sort()
    assert (
        merge_hyperedge_list ==
        gain_manager.distribution.circuit.hyperedge_list
    )

    split_gain = gain_manager.split_hyperedge_gain(
        old_hyperedge=old_hyperedge,
        new_hyperedge_list=[
            to_merge_hyperedge_one,
            to_merge_hyperedge_two
        ]
    )
    assert split_gain == -merge_gain


def test_merge_with_embedding():

    network = NISQNetwork(
        server_coupling=[[0, 1], [0, 2], [0, 3]],
        server_qubits={0: [0], 1: [1], 2: [2], 3: [3]}
    )

    circ = Circuit(3)
    circ.add_gate(OpType.CU1, 1.0, [0, 2])
    circ.add_gate(OpType.CU1, 1.0, [0, 1])
    circ.H(0)
    circ.add_gate(OpType.CU1, 1.0, [0, 2])
    circ.H(0)
    circ.add_gate(OpType.CU1, 1.0, [0, 2])

    hyp_circ = HypergraphCircuit(circ)

    placement = Placement({0: 1, 1: 2, 2: 3, 3: 3, 4: 2, 5: 3, 6: 3})

    distribution = Distribution(
        circuit=hyp_circ,
        placement=placement,
        network=network
    )
    assert distribution.cost() == 7

    gain_mgr = GainManager(initial_distribution=distribution)

    # Here we are merging either side of the 'CX' to embed it.
    # Note that we can make use of the connection that remains after
    # embedding to implement the last CZ
    to_merge_hyperedge_list = [
        Hyperedge(vertices=[0, 3, 4], weight=1),
        Hyperedge(vertices=[0, 6], weight=1)
    ]
    assert gain_mgr.merge_hyperedge_gain(
        to_merge_hyperedge_list=to_merge_hyperedge_list
    ) == 2
    gain_mgr.merge_hyperedge(to_merge_hyperedge_list=to_merge_hyperedge_list)
    assert gain_mgr.distribution.cost() == 5
