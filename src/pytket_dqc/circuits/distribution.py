from pytket_dqc.circuits.hypergraph_circuit import HypergraphCircuit
from pytket_dqc.circuits.hypergraph import Hypergraph
from pytket_dqc.placement import Placement
from pytket_dqc.networks import NISQNetwork
from pytket import Circuit


class Distribution():

    def __init__(
        self,
        circuit: HypergraphCircuit,
        packets: Hypergraph,
        placement: Placement,
    ):

        self.circuit = circuit
        self.packets = packets
        self.placement = placement

    def to_pytket_circuit(self, network: NISQNetwork) -> Circuit:
        raise NotImplementedError
