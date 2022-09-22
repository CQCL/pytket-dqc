from pytket_dqc.circuits.hypergraph_circuit import HypergraphCircuit
from pytket_dqc.placement import Placement
from pytket_dqc.networks import NISQNetwork
from pytket import Circuit


class Distribution:
    """Class containing all information required to generate pytket circuit.
    """

    def __init__(
        self,
        circuit: HypergraphCircuit,
        placement: Placement,
        network: NISQNetwork,
    ):
        """Initialisation function for Distribution

        :param circuit: Circuit to be distributed, including its hypergraph.
        :type circuit: HypergraphCircuit
        :param placement: A placement of the qubits and gates onto servers.
        :type placement: Placement
        :param network: Network onto which circuit is distributed.
        :type network: NISQNetwork
        :raises Exception: Raised if the placement is not valid for the
            circuit.
        :raises Exception: Raised if the placement is not valid for the
            packets.
        """

        self.circuit = circuit
        self.placement = placement
        self.network = network

    def is_valid(self) -> bool:
        """Check that this distribution can be implemented.
        """

        # TODO: There may be some other checks that we want to do here to check
        # that the packets hypergraph is not totally nonsensical. For example
        # that gate nodes are not in too many packets. I think the conditions
        # that are most relevant will emerge as the to_pytket_circuit method
        # is written.
        return self.placement.is_valid(
            self.circuit, self.network
        )

    def cost(self) -> int:
        """Return the number of ebits required for this distribution.
        """

        # TODO: Once the ALAP algorithm is introduced, this will be replaced
        # by a for loop over all hyperedges, calling the function to calculate
        # their cost, which uses GainManager.
        return self.placement.cost(self.circuit, self.network)

    def to_pytket_circuit(self) -> Circuit:
        if self.is_valid():
            raise Exception("The distribution of the circuit is not valid!")
        raise NotImplementedError
