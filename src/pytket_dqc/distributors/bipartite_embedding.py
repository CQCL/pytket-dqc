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


class CoverEmbedding(Distributor):
    """ Distributor applying :class:`.VertexCover` refinement to an
    initial distribution. This workflow
    is the simplest one considering embedding in the first instance.
    """

    def distribute(
        self, circ: Circuit, network: NISQNetwork, **kwargs
    ) -> Distribution:
        """Method producing a distribution of the given circuit
        onto the given network.
        
        Note that kwargs are passed on to
        :class:`.VertexCover` and the `distribute` method of the initial
        distributor.

        :param circ: Circuit to be distributed
        :type circ: Circuit
        :param network: Network onto which circuit should be distributed
        :type network: NISQNetwork
        :return: Distribution of circ onto network.
        :rtype: Distribution

        :key initial_distributor: Initial distributor to be used to
            generate distribution refined by :class:`.VertexCover`.
            Default is :class:`PartitioningHeterogeneous`
        """

        initial_distributor = kwargs.get(
            'initial_distributor', PartitioningHeterogeneous()
        )
        distribution = initial_distributor.distribute(
            circ, network, **kwargs
        )
        VertexCover().refine(distribution, **kwargs)

        return distribution


class CoverEmbeddingSteiner(Distributor):
    """ Distributor refining the output of :class:`.CoverEmbedding`
    to make use of Steiner trees.
    """

    def distribute(
        self, circ: Circuit, network: NISQNetwork, **kwargs
    ) -> Distribution:
        """Abstract method producing a distribution of the given circuit
        onto the given network.
        
        Note that kwargs are passed on to
        the `distribute` method of :class:`.CoverEmbedding`.

        :param circ: Circuit to be distributed
        :type circ: Circuit
        :param network: Network onto which circuit should be distributed
        :type network: NISQNetwork
        :return: Distribution of circ onto network.
        :rtype: Distribution
        """

        refiner_list = [
            NeighbouringDTypeMerge(),
            IntertwinedDTypeMerge(),
        ]
        refiner = RepeatRefiner(SequenceRefiner(refiner_list))

        distribution = CoverEmbedding().distribute(
            circ, network, **kwargs
        )
        refiner.refine(distribution)

        return distribution


class CoverEmbeddingSteinerDetached(Distributor):
    """ Distributor refining  the output of
    :class:`.BipartiteEmbeddingSteiner` to make use of detached gates.
    """

    def distribute(
        self, circ: Circuit, network: NISQNetwork, **kwargs
    ) -> Distribution:
        """Abstract method producing a distribution of the given circuit
        onto the given network.
        
        Note that kwargs are passed on to
        the `distribute` method of :class:`.CoverEmbeddingSteiner`. and
        the refine method of :class:`.DetachedGates`

        :param circ: Circuit to be distributed
        :type circ: Circuit
        :param network: Network onto which circuit should be distributed
        :type network: NISQNetwork
        :return: Distribution of circ onto network.
        :rtype: Distribution
        """

        distribution = CoverEmbeddingSteiner().distribute(
            circ, network, **kwargs
        )
        DetachedGates().refine(
            distribution,
            **kwargs,
        )

        return distribution
