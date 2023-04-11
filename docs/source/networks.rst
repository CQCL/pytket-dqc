Networks
========

.. autoclass:: pytket_dqc.networks.server_network.ServerNetwork

    .. automethod:: ServerNetwork.__init__
    
    .. automethod:: ServerNetwork.is_placement

    .. automethod:: ServerNetwork.get_server_list

    .. automethod:: ServerNetwork.draw_server_network

.. autoclass:: pytket_dqc.networks.nisq_network.HeterogeneousNetwork

    .. automethod:: HeterogeneousNetwork.__init__

    .. automethod:: HeterogeneousNetwork.get_qubit_list

    .. automethod:: HeterogeneousNetwork.draw_nisq_network

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