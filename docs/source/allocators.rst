Allocators
============

.. autoclass:: pytket_dqc.allocators.Allocator

    .. automethod:: Allocator.__init__

    .. automethod:: Allocator.allocate

.. autoclass:: pytket_dqc.allocators.ordered.Ordered

    .. automethod:: Ordered.__init__

    .. automethod:: Ordered.allocate

.. autofunction:: pytket_dqc.allocators.ordered.order_reducing_size

.. autoclass:: pytket_dqc.allocators.annealing.Annealing

    .. automethod:: Annealing.__init__

    .. automethod:: Annealing.allocate

.. autoclass:: pytket_dqc.allocators.brute.Brute

    .. automethod:: Brute.__init__

    .. automethod:: Brute.allocate

.. autoclass:: pytket_dqc.allocators.hypergraph_partitioning.HypergraphPartitioning

    .. automethod:: HypergraphPartitioning.__init__

    .. automethod:: HypergraphPartitioning.allocate

.. autoclass:: pytket_dqc.allocators.random.Random

    .. automethod:: Random.__init__

    .. automethod:: Random.allocate

.. autoclass:: pytket_dqc.allocators.routing.Routing

    .. automethod:: Routing.__init__

    .. automethod:: Routing.allocate