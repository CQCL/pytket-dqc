from pytket_dqc import DistributedCircuit, Hypergraph
from pytket import Circuit


def test_simple():

    circ = Circuit(2).CZ(0, 1)
    dist_circ = DistributedCircuit(circ)
    # dist_circ.from_circuit()
    assert dist_circ.get_circuit() == circ

    hypgra = dist_circ.get_hypergraph()
    assert isinstance(hypgra, Hypergraph)
