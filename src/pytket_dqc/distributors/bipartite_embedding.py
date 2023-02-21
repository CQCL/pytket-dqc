from pytket_dqc.refiners import (
    VertexCover,
)
from pytket_dqc.allocators import HypergraphPartitioning
from pytket import Circuit
from .distributor import Distributor
from pytket_dqc import NISQNetwork, Distribution
from pytket_dqc.refiners import (
    NeighbouringDTypeMerge,
    IntertwinedDTypeMerge,
    SequenceRefiner,
    RepeatRefiner,
    BoundaryReallocation,
)


class BipartiteEmbedding(Distributor):

    def distribute(
        self, circ: Circuit, network: NISQNetwork, **kwargs
    ) -> Distribution:

        seed = kwargs.get('seed', None)

        distribution = HypergraphPartitioning().allocate(
            circ, network, seed=seed, num_rounds=0
        )
        VertexCover().refine(distribution, vertex_cover_alg='networkx')

        return distribution


class BipartiteEmbeddingSteiner(Distributor):

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

    def distribute(
        self, circ: Circuit, network: NISQNetwork, **kwargs
    ) -> Distribution:

        distribution = BipartiteEmbeddingSteiner().distribute(
            circ, network, **kwargs
        )
        BoundaryReallocation().refine(
            distribution,
            reallocate_qubits=False,
            **kwargs,
        )

        return distribution
