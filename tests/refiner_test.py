import pytest
from pytket import Circuit, OpType  # type: ignore
from pytket_dqc import Distribution
from pytket_dqc.placement import Placement
from pytket_dqc.utils import check_equivalence
from pytket_dqc.networks import NISQNetwork

def test_embedding_and_not_embedding():

    network = NISQNetwork(
        server_coupling=[[0, 1], [1, 2], [1, 3]],
        server_qubits={0: [0], 1: [1], 2: [2], 3: [3]}
    )

    circ = Circuit(4)

    # These gates will be in different hyperedges
    circ.add_gate(OpType.CU1, 1.0, [0, 2])
    circ.Rz(0.3, 0)
    circ.add_gate(OpType.CU1, 1.0, [0, 1])

    # Will be embedded
    circ.H(2)
    circ.add_gate(OpType.CU1, 1.0, [3, 2])
    circ.H(2)

    # Allows for embedding, but will not be
    circ.H(0)
    circ.add_gate(OpType.CU1, 1.0, [0, 2])
    circ.add_gate(OpType.CU1, 1.0, [3, 0])
    circ.H(0)

    circ.add_gate(OpType.CU1, 1.0, [1, 0])

    hyp_circ = HypergraphCircuit(circ)

    # Empty all dictionaries
    hyp_circ.hyperedge_list = []
    hyp_circ.hyperedge_dict = {v: [] for v in hyp_circ.vertex_list}
    hyp_circ.vertex_neighbours = {v: set() for v in hyp_circ.vertex_list}

    hyp_circ.add_hyperedge([0, 4])
    hyp_circ.add_hyperedge([0, 5])
    hyp_circ.add_hyperedge([0, 7, 8])
    hyp_circ.add_hyperedge([0, 9])
    hyp_circ.add_hyperedge([1, 5, 9])
    hyp_circ.add_hyperedge([2, 4, 7])   # Merged hyperedge
    hyp_circ.add_hyperedge([2, 6])
    hyp_circ.add_hyperedge([3, 6, 8])

    placement = Placement({0: 0, 1: 1, 2: 2, 3: 3, 4: 2,
                          5: 1, 6: 2, 7: 2, 8: 1, 9: 0})

    distribution = Distribution(
        circuit=hyp_circ, placement=placement, network=network)

    pytket_circ = distribution.to_pytket_circuit()

    check_equivalence(
        circ, pytket_circ, distribution.get_qubit_mapping()
    )