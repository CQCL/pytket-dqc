Data structures
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

    .. automethod:: Hypergraph.to_dict

    .. automethod:: Hypergraph.from_dict

.. autoclass:: pytket_dqc.circuits.hypergraph_circuit.HypergraphCircuit
    
    .. automethod:: HypergraphCircuit.__init__

    .. automethod:: HypergraphCircuit.add_hyperedge

    .. automethod:: HypergraphCircuit.to_dict

    .. automethod:: HypergraphCircuit.from_dict

.. autoclass:: pytket_dqc.circuits.distribution.Distribution
    
    .. automethod:: Distribution.__init__

    .. automethod:: Distribution.cost

    .. automethod:: Distribution.non_local_gate_count

    .. automethod:: Distribution.detached_gate_count

    .. automethod:: Distribution.to_pytket_circuit

    .. automethod:: Distribution.get_qubit_mapping

    .. automethod:: Distribution.to_dict

    .. automethod:: Distribution.from_dict
