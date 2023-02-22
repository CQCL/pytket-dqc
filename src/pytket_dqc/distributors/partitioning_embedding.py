from .distributor import Distributor
from pytket_dqc import NISQNetwork, Distribution
from pytket_dqc.refiners import (
    RepeatRefiner,
    EagerHTypeMerge,
    BoundaryReallocation,
)
from pytket_dqc.allocators import HypergraphPartitioning
from pytket import Circuit


class PartitioningHeterogeneous(Distributor):
    """ Distributor refining the output of :class:`.HypergraphPartitioning`
    to adapt the result to heterogeneous networks.
    """

    def distribute(
        self, circ: Circuit, network: NISQNetwork, **kwargs
    ) -> Distribution:

        distribution = HypergraphPartitioning().allocate(
            circ, network, **kwargs
        )
        refiner = BoundaryReallocation(**kwargs)
        refiner.refine(distribution)

        return distribution


class PartitioningHeterogeneousEmbedding(Distributor):
    """ Distributor refining the output of :class:`.PartitioningHeterogeneous`
    to make use of embedding.
    """

    def distribute(
        self, circ: Circuit, network: NISQNetwork, **kwargs
    ) -> Distribution:

        distribution = PartitioningHeterogeneous().distribute(
            circ, network, **kwargs
        )
        refiner = RepeatRefiner(EagerHTypeMerge())
        refiner.refine(distribution)

        return distribution
