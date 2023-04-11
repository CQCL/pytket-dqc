Networks
========

.. autoclass:: pytket_dqc.networks.module_network.ModuleNetwork

    .. automethod:: ModuleNetwork.__init__
    
    .. automethod:: ModuleNetwork.is_placement

    .. automethod:: ModuleNetwork.get_server_list

    .. automethod:: ModuleNetwork.draw_module_network

.. autoclass:: pytket_dqc.networks.nisq_network.HeterogeneousNetwork

    .. automethod:: HeterogeneousNetwork.__init__

    .. automethod:: HeterogeneousNetwork.get_qubit_list

    .. automethod:: HeterogeneousNetwork.draw_heterogeneous_network

    .. automethod:: HeterogeneousNetwork.to_dict

    .. automethod:: HeterogeneousNetwork.from_dict

.. autoclass:: pytket_dqc.networks.nisq_network.AllToAll

    .. automethod:: AllToAll.__init__

.. autoclass:: pytket_dqc.networks.random_networks.ScaleFreeNetwork

    .. automethod:: ScaleFreeNetwork.__init__

.. autoclass:: pytket_dqc.networks.random_networks.SmallWorldNetwork

    .. automethod:: SmallWorldNetwork.__init__

.. autoclass:: pytket_dqc.networks.random_networks.RandomNetwork

    .. automethod:: RandomNetwork.__init__