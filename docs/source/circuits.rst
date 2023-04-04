Circuits
============

.. autoclass:: pytket_dqc.circuits.hypergraph.Hypergraph

    .. automethod:: Hypergraph.__init__

    .. automethod:: Hypergraph.is_placement

    .. automethod:: Hypergraph.draw

    .. automethod:: Hypergraph.add_vertices

    .. automethod:: Hypergraph.add_hyperedge

    .. automethod:: Hypergraph.merge_hyperedge
    
    .. automethod:: Hypergraph.split_hyperedge

    .. automethod:: Hypergraph.remove_hyperedge

.. autoclass:: pytket_dqc.circuits.hypergraph_circuit.HypergraphCircuit
    
    .. automethod:: HypergraphCircuit.__init__

    .. automethod:: HypergraphCircuit.add_hyperedge

.. autoclass:: pytket_dqc.circuits.distribution.Distribution
    
    .. automethod:: Distribution.__init__

    .. automethod:: Distribution.cost

    .. automethod:: Distribution.to_pytket_circuit

    .. automethod:: Distribution.get_qubit_mapping
