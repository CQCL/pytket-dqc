from .distributor import Distributor
from pytket_dqc import NISQNetwork, Distribution
from pytket_dqc.refiners import (
    RepeatRefiner,
    EagerHTypeMerge,
)
from pytket_dqc.allocators import HypergraphPartitioning
from pytket import Circuit


class PartitioningEmbedding(Distributor):
    """ Distributor refining the output of :class:`.HypergraphPartitioning`
    to make use of embedding.
    """

    def distribute(
        self, circ: Circuit, network: NISQNetwork, **kwargs
    ) -> Distribution:

        distribution = HypergraphPartitioning().allocate(
            circ, network, **kwargs
        )
        refiner = RepeatRefiner(EagerHTypeMerge())
        refiner.refine(distribution)

        return distribution
