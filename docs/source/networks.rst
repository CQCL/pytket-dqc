Networks
========

.. autoclass:: pytket_dqc.networks.server_network.ServerNetwork

    .. automethod:: ServerNetwork.__init__
    
    .. automethod:: ServerNetwork.is_placement

    .. automethod:: ServerNetwork.get_server_list

    .. automethod:: ServerNetwork.draw_server_network

.. autoclass:: pytket_dqc.networks.nisq_network.NISQNetwork

    .. automethod:: NISQNetwork.__init__

    .. automethod:: NISQNetwork.get_qubit_list

    .. automethod:: NISQNetwork.draw_nisq_network

    .. automethod:: NISQNetwork.to_dict

    .. automethod:: NISQNetwork.from_dict

.. autoclass:: pytket_dqc.networks.nisq_network.AllToAll

    .. automethod:: AllToAll.__init__

.. autoclass:: pytket_dqc.networks.random_networks.ScaleFreeNISQNetwork

    .. automethod:: ScaleFreeNISQNetwork.__init__

.. autoclass:: pytket_dqc.networks.random_networks.SmallWorldNISQNetwork

    .. automethod:: SmallWorldNISQNetwork.__init__

.. autoclass:: pytket_dqc.networks.random_networks.RandomNISQNetwork

    .. automethod:: RandomNISQNetwork.__init__