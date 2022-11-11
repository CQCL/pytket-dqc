from pytket_dqc.networks import NISQNetwork
from pytket_dqc import HypergraphCircuit, Distribution
from pytket_dqc.circuits import Hyperedge
from pytket_dqc.placement import Placement
from pytket import Circuit, OpType


# TODO: Add tests with circuits where one or more qubits are unused


def test_distribution_valid():

    large_network = NISQNetwork(
        [[0, 1], [0, 2], [1, 2]], {0: [0, 1, 2], 1: [3, 4, 5], 2: [6, 7, 8, 9]}
    )
    small_network = NISQNetwork([[0, 1]], {0: [0, 1], 1: [2]})

    small_circ = Circuit(2).add_gate(OpType.CU1, 1.0, [0, 1])
    dist_small_circ = HypergraphCircuit(small_circ)

    med_circ = (
        Circuit(4)
        .add_gate(OpType.CU1, 1.0, [0, 1])
        .add_gate(OpType.CU1, 1.0, [1, 2])
        .add_gate(OpType.CU1, 1.0, [2, 3])
    )
    dist_med_circ = HypergraphCircuit(med_circ)

    placement_one = Placement({0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1})
    placement_two = Placement({0: 1, 1: 1, 2: 1, 3: 0, 4: 0, 5: 0, 6: 2})
    placement_three = Placement({0: 2, 1: 2, 2: 2, 3: 2, 4: 0, 5: 0, 6: 0})
    placement_four = Placement({0: 0, 1: 0, 2: 0, 3: 2, 4: 0, 5: 0, 6: 0})
    placement_five = Placement({0: 0, 1: 0, 2: 0, 3: 0})
    placement_six = Placement({0: 2, 1: 2, 2: 3})
    placement_seven = Placement({0: 1, 1: 1, 2: 1, 3: 0, 4: 1, 5: 1, 6: 1})
    placement_eight = Placement({0: 1, 1: 0, 2: 1})
    placement_nine = Placement({0: 2, 1: 2, 2: 2})

    assert not Distribution(
        dist_med_circ, placement_one, large_network
    ).is_valid()
    assert Distribution(dist_med_circ, placement_two, large_network).is_valid()
    assert Distribution(
        dist_med_circ, placement_three, large_network
    ).is_valid()
    assert Distribution(
        dist_med_circ, placement_four, large_network
    ).is_valid()
    assert not Distribution(
        dist_med_circ, placement_five, large_network
    ).is_valid()
    assert not Distribution(
        dist_med_circ, placement_six, large_network
    ).is_valid()
    assert not Distribution(
        dist_small_circ, placement_seven, small_network
    ).is_valid()
    assert Distribution(
        dist_small_circ, placement_eight, small_network
    ).is_valid()
    assert not Distribution(
        dist_small_circ, placement_nine, small_network
    ).is_valid()


def test_distribution_cost_no_embedding():

    two_CZ_circ = (
        Circuit(3)
        .add_gate(OpType.CU1, 1.0, [0, 1])
        .add_gate(OpType.CU1, 1.0, [0, 2])
    )
    dist_two_CZ_circ = HypergraphCircuit(two_CZ_circ)

    three_line_network = NISQNetwork(
        [[0, 1], [1, 2], [1, 3], [2, 4]],
        {0: [0], 1: [1], 2: [2], 3: [3], 4: [4]},
    )

    placement_one = Placement({0: 0, 1: 1, 2: 2, 3: 1, 4: 2})
    assert (
        Distribution(
            dist_two_CZ_circ, placement_one, three_line_network
        ).cost()
        == 2
    )
    placement_two = Placement({0: 0, 1: 1, 2: 2, 3: 1, 4: 0})
    assert (
        Distribution(
            dist_two_CZ_circ, placement_two, three_line_network
        ).cost()
        == 3
    )
    placement_three = Placement({0: 1, 1: 0, 2: 2, 3: 0, 4: 2})
    assert (
        Distribution(
            dist_two_CZ_circ, placement_three, three_line_network
        ).cost()
        == 2
    )


def test_alap():

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

    placement = Placement(
        {
            0: 1,
            1: 1,
            2: 4,
            3: 4,
            4: 1,
            5: 2,
            6: 1,
            7: 4,
            8: 4,
            9: 4,
            10: 0,
            11: 1,
            12: 4,
            13: 2,
        }
    )

    distribution = Distribution(HypergraphCircuit(circ), placement, network)

    hyp_0 = Hyperedge([0, 5, 7, 11, 12])
    hyp_2 = Hyperedge([2, 4, 5, 6, 13])

    # These costs have been calculated by hand.
    # The examples attempted are meant to be interesting cases but this
    # is not in any way exhaustive.
    assert distribution.hyperedge_cost(hyp_0) == 4
    assert distribution.hyperedge_cost(hyp_2) == 6

    distribution.placement.placement[2] = 0
    assert distribution.hyperedge_cost(hyp_2) == 3

    distribution.placement.placement[2] = 0
    distribution.placement.placement[13] = 1
    assert distribution.hyperedge_cost(hyp_2) == 2


def test_alap_on_hyperedge_requiring_euler():
    # The circuit given below has a hyperedge between the first and last
    # CU1 gates. The gates in between can all be embedded but, to do so,
    # it is required to decompose the middle Hadamard to its Euler form
    # and squash the Rz(0.5) accordingly.

    circ = Circuit(5)
    circ.add_gate(OpType.CU1, [0.3], [0, 1])
    circ.H(0)
    circ.Rz(0.5, 0)
    circ.add_gate(OpType.CU1, [1.0], [0, 3])
    circ.H(0)
    circ.add_gate(OpType.CU1, [1.0], [0, 4])
    circ.Rz(0.5, 0)
    circ.H(0)
    circ.add_gate(OpType.CU1, [0.8], [0, 2])

    network = NISQNetwork([[0, 1]], {0: [0], 1: [1, 2, 3, 4]},)

    placement = Placement(
        {0: 0, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1}
    )

    distribution = Distribution(HypergraphCircuit(circ), placement, network)
    assert distribution.is_valid()

    hyp = Hyperedge([0, 5, 8])

    # This cost has been calculated by hand
    assert distribution.hyperedge_cost(hyp) == 1


def test_alap_on_hyperedge_mixing_H_and_D_embeddings():

    circ = Circuit(4)
    circ.add_gate(OpType.CU1, 0.1234, [0, 1])  # Gate 4
    circ.H(1)
    circ.add_gate(OpType.CU1, 1.0, [1, 2])  # Gate 5, H-embedded
    circ.add_gate(OpType.CU1, 1.0, [1, 3])  # Gate 6, H-embedded
    circ.H(1)
    circ.add_gate(OpType.CU1, 0.1234, [1, 3])  # Gate 7, D-embedded
    circ.add_gate(OpType.CU1, 0.1234, [1, 2])  # Gate 8

    network = NISQNetwork([[0, 1], [1, 2]], {0: [0], 1: [1], 2: [2, 3]})

    placement = Placement(
        {0: 0, 1: 1, 2: 2, 3: 2, 4: 0, 5: 1, 6: 1, 7: 1, 8: 2}
    )

    distribution = Distribution(HypergraphCircuit(circ), placement, network)
    assert distribution.is_valid()

    hyp = Hyperedge([1, 4, 8])
    assert distribution.circuit.requires_h_embedded_cu1(hyp)

    # This cost has been calculated by hand
    assert distribution.hyperedge_cost(hyp) == 2


def test_distribution_cost_with_embedding():
    # Note: This is testing ALAP as well

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

    placement = Placement(
        {
            0: 1,
            1: 1,
            2: 2,
            3: 4,
            4: 1,
            5: 2,
            6: 4,
            7: 4,
            8: 2,
            9: 4,
            10: 0,
            11: 1,
            12: 4,
            13: 3,
        }
    )

    dist_circ = HypergraphCircuit(circ)
    # Reset the hyperedges
    dist_circ.hyperedge_list = []
    for vertex in dist_circ.vertex_list:
        dist_circ.hyperedge_dict[vertex] = []
        dist_circ.vertex_neighbours[vertex] = set()
    # New hyperedges: some are the same, some are merged
    new_hyperedges = []
    new_hyperedges.append(Hyperedge([0, 11, 12]))
    new_hyperedges.append(Hyperedge([0, 5, 7]))
    new_hyperedges.append(Hyperedge([0, 8, 9]))
    new_hyperedges.append(Hyperedge([1, 4, 10, 11, 13]))
    new_hyperedges.append(Hyperedge([2, 4, 5, 6, 13]))  # Merged hyperedge
    new_hyperedges.append(Hyperedge([2, 8, 10]))
    new_hyperedges.append(Hyperedge([3, 6, 7, 9, 12]))  # Merged hyperedge
    # Add the new hyperedges
    for hyperedge in new_hyperedges:
        dist_circ.add_hyperedge(hyperedge.vertices)

    distribution = Distribution(dist_circ, placement, network)

    assert not dist_circ.requires_h_embedded_cu1(new_hyperedges[0])
    assert distribution.hyperedge_cost(new_hyperedges[0]) == 3
    assert not dist_circ.requires_h_embedded_cu1(new_hyperedges[1])
    assert distribution.hyperedge_cost(new_hyperedges[1]) == 4
    assert not dist_circ.requires_h_embedded_cu1(new_hyperedges[2])
    assert distribution.hyperedge_cost(new_hyperedges[2]) == 4
    assert not dist_circ.requires_h_embedded_cu1(new_hyperedges[3])
    assert distribution.hyperedge_cost(new_hyperedges[3]) == 2
    assert dist_circ.requires_h_embedded_cu1(new_hyperedges[4])
    assert distribution.hyperedge_cost(new_hyperedges[4]) == 6
    assert not dist_circ.requires_h_embedded_cu1(new_hyperedges[5])
    assert distribution.hyperedge_cost(new_hyperedges[5]) == 1
    assert not dist_circ.requires_h_embedded_cu1(new_hyperedges[6])
    assert distribution.hyperedge_cost(new_hyperedges[6]) == 0

    assert distribution.cost() == 3 + 4 + 4 + 2 + 6 + 1 + 0
    assert distribution.is_valid()
