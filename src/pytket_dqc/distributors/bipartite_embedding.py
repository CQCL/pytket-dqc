from pytket_dqc.refiners import (
    VertexCover,
)
from pytket import Circuit
from .distributor import Distributor
from .partitioning_embedding import PartitioningHeterogeneous
from pytket_dqc import NISQNetwork, Distribution
from pytket_dqc.refiners import (
    NeighbouringDTypeMerge,
    IntertwinedDTypeMerge,
    SequenceRefiner,
    RepeatRefiner,
    DetachedGates,
)


class BipartiteEmbedding(Distributor):
    """ Distributor applying :class:`.VertexCover` refinement to an
    initial allocation by :class:`.HypergraphPartitioning`. This workflow
    is the simplest one considering embedding in the first instance.
    """

    def distribute(
        self, circ: Circuit, network: NISQNetwork, **kwargs
    ) -> Distribution:

        distribution = PartitioningHeterogeneous().distribute(
            circ, network, **kwargs
        )
        VertexCover().refine(distribution, **kwargs)

        return distribution


class BipartiteEmbeddingSteiner(Distributor):
    """ Distributor refining the output of :class:`.BipartiteEmbedding`
    to make use of Steiner trees.
    """

    def distribute(
        self, circ: Circuit, network: NISQNetwork, **kwargs
    ) -> Distribution:

        refiner_list = [
            NeighbouringDTypeMerge(),
            IntertwinedDTypeMerge(),
        ]
        refiner = RepeatRefiner(SequenceRefiner(refiner_list))

        distribution = BipartiteEmbedding().distribute(
            circ, network, **kwargs
        )
        refiner.refine(distribution)

        return distribution


class BipartiteEmbeddingSteinerDetached(Distributor):
    """ Distributor refining  the output of
    :class:`.BipartiteEmbeddingSteiner` to make use of detached gates.
    """

    def distribute(
        self, circ: Circuit, network: NISQNetwork, **kwargs
    ) -> Distribution:

        distribution = BipartiteEmbeddingSteiner().distribute(
            circ, network, **kwargs
        )
        DetachedGates().refine(
            distribution,
            **kwargs,
        )

        return distribution
