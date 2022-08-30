from pytket_dqc.circuits.hypergraph_circuit import HypergraphCircuit
from pytket_dqc.circuits.hypergraph import Hypergraph
from pytket_dqc.placement import Placement
from pytket_dqc.networks import NISQNetwork
from pytket import Circuit


class Distribution():
    """Class containing all information required to generate pytket circuit.
    """

    def __init__(
        self,
        circuit: HypergraphCircuit,
        packets: Hypergraph,
        placement: Placement,
    ):
        """Initialisation function for Distribution

        :param circuit: Circuit which is to be distributed
        :type circuit: HypergraphCircuit
        :param packets: A description of the packets within the circuit.
        These are those gates which can be implemented with a single e-bit.
        :type packets: Hypergraph
        :param placement: A placement of the qubits and gates onto servers.
        :type placement: Placement
        :raises Exception: Raised if the placement is not valid for the
        circuit.
        :raises Exception: Raised if the placement is not valid for the
        packets.
        """

        if not circuit.is_placement(placement):
            raise Exception("This placement is not valid for circuit")

        if not packets.is_placement(placement):
            raise Exception("This placement is not valid for packets")

        # TODO: There may be some other checks that we want to do here to check
        # that the packets hypergraph is not totally nonsensical. For example
        # that gate nodes are not in too many packets. I think the conditions
        # that are most relevant will emerge as the to_pytket_circuit method
        # is written.

        self.circuit = circuit
        self.packets = packets
        self.placement = placement

    def to_pytket_circuit(self, network: NISQNetwork) -> Circuit:
        raise NotImplementedError
