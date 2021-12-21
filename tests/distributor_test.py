from pytket_dqc.distributors import Annealing
from pytket_dqc import DistributedCircuit
from pytket import Circuit


def test_simple():

    circ = Circuit(2).CZ(0, 1)
    dist_circ = DistributedCircuit(circ)
    # dist_circ.from_circuit(circ)
    distributor = Annealing()
    assert isinstance(distributor.distribute(dist_circ), dict)
