from .distributor import Distributor
from pytket_dqc import HeterogeneousNetwork, Distribution
from pytket_dqc.refiners import (
    RepeatRefiner,
    EagerHTypeMerge,
    BoundaryReallocation,
)
from pytket_dqc.allocators import HypergraphPartitioning, Annealing
from pytket import Circuit


class PartitioningAnnealing(Distributor):
    """ Distributor using the :class:`.Annealing` allocator.
    """

    def distribute(
        self, circ: Circuit, network: HeterogeneousNetwork, **kwargs
    ) -> Distribution:
        """Method producing a distribution of the given circuit
        onto the given network.

        Note that kwargs are passed on to the allocate method of
        :class:`.Annealing`.

        :param circ: Circuit to be distributed
        :type circ: Circuit
        :param network: Network onto which circuit should be distributed
        :type network: HeterogeneousNetwork
        :return: Distribution of circ onto network.
        :rtype: Distribution
        """

        return Annealing().allocate(
            circ, network, **kwargs
        )


class PartitioningHeterogeneous(Distributor):
    """ Distributor refining the output of :class:`.HypergraphPartitioning`
    to adapt the result to heterogeneous networks.
    """

    def distribute(
        self, circ: Circuit, network: HeterogeneousNetwork, **kwargs
    ) -> Distribution:
        """Method producing a distribution of the given circuit
        onto the given network.

        Note that kwargs are passed on to the allocate method of
        :class:`.HypergraphPartitioning` and the `refine` method of
        :class:`.BoundaryReallocation`.

        :param circ: Circuit to be distributed
        :type circ: Circuit
        :param network: Network onto which circuit should be distributed
        :type network: HeterogeneousNetwork
        :return: Distribution of circ onto network.
        :rtype: Distribution
        """

        distribution = HypergraphPartitioning().allocate(
            circ, network, **kwargs
        )
        refiner = BoundaryReallocation()
        refiner.refine(distribution, **kwargs)

        return distribution


class PartitioningHeterogeneousEmbedding(Distributor):
    """ Distributor refining the output of :class:`.PartitioningHeterogeneous`
    to make use of embedding.
    """

    def distribute(
        self, circ: Circuit, network: HeterogeneousNetwork, **kwargs
    ) -> Distribution:
        """Method producing a distribution of the given circuit
        onto the given network.

        Note that kwargs are passed on to the distribute method of
        :class:`.PartitioningHeterogeneousEmbedding`.

        :param circ: Circuit to be distributed
        :type circ: Circuit
        :param network: Network onto which circuit should be distributed
        :type network: HeterogeneousNetwork
        :return: Distribution of circ onto network.
        :rtype: Distribution
        """

        initial_distributor = kwargs.get(
            'initial_distributor', PartitioningHeterogeneous()
        )

        distribution = initial_distributor.distribute(
            circ, network, **kwargs
        )
        refiner = RepeatRefiner(EagerHTypeMerge())
        refiner.refine(distribution)

        return distribution
