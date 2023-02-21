from .nisq_network import NISQNetwork
import random
import networkx as nx  # type:ignore
import math
import numpy as np


class RandomNISQNetwork(NISQNetwork):
    """NISQNetwok with underlying server network that is random but connected.
    In particular graphs are erdos renyi random graphs, post selected
    on graphs that are connected. The ebit memory for each module is the the
    larger of 2 and largest integer less than the average number of qubits
    in each module.
    """

    def __init__(self, n_servers: int, n_qubits: int, **kwargs):
        """Initialisation function.

        :param n_servers: The number of servers.
        :type n_servers: int
        :param n_qubits: The total number of qubits.
        :type n_qubits: int
        :raises Exception: Raised if the number of qubits is less than the
            number of servers.

        :key edge_prob: The probability of an edge between two servers,
            defaults to 2/(n_servers-1), adding roughly 2 edges per server.
        """

        if n_qubits < n_servers:
            raise Exception(
                "The number of qubits must be greater ",
                "than the number of servers.")

        edge_prob = kwargs.get("edge_prob", 1/math.factorial(n_servers-2))
        seed = kwargs.get("seed", None)
        np.random.seed(seed)

        # Generate erdos renyi graph until one that is connected is generated.
        connected = False
        while not connected:
            graph = nx.gnp_random_graph(
                n=n_servers,
                p=edge_prob,
                seed=np.random,
            )
            connected = nx.is_connected(graph)
        server_coupling = [list(edge) for edge in graph.edges]

        # Assign at least one qubit to each server
        server_qubits = {i: [i] for i in range(n_servers)}

        # Assign remaining qubits randomly.
        random.seed(seed)
        for qubit in range(n_servers, n_qubits):
            server = random.randrange(n_servers)
            server_qubits[server] = server_qubits[server] + [qubit]

        server_ebit_mem = {
            server: max((n_qubits-1)//(n_servers), 2)
            for server in server_qubits.keys()
        }

        super().__init__(
            server_coupling=server_coupling,
            server_qubits=server_qubits,
            server_ebit_mem=server_ebit_mem,
        )


class ScaleFreeNISQNetwork(NISQNetwork):
    """NISQNetwork with underlying server network that is scale-free. This is
    to say one whose degree distribution follows a power law. The ebit
    memory for each module is the the larger of 2 and largest integer less
    than the average number of qubits in each module.
    """

    def __init__(self, n_servers: int, n_qubits: int, **kwargs):
        """Initialisation method for scale-free network.

        :param n_servers: The number of servers in the network.
        :type n_servers: int
        :param n_qubits: The total number of qubits. Qubits are assigned
            randomly to each server, with at least one per server.
        :type n_qubits: int
        :raises Exception: Raised if the number of qubits is less than the
            number of servers.
        """

        if n_qubits < n_servers:
            raise Exception(
                "The number of qubits must be greater ",
                "than the number of servers."
            )

        m = kwargs.get('m', 1)
        seed = kwargs.get('seed', None)
        initial_graph = kwargs.get('initial_graph', None)

        # Generate barabasi albert graph
        graph = nx.barabasi_albert_graph(
            n=n_servers,
            m=m,
            seed=seed,
            initial_graph=initial_graph
        )
        server_coupling = [list(edge) for edge in graph.edges]

        random.seed(seed)

        # Assign at least one qubit to each server
        server_qubits = {i: [i] for i in range(n_servers)}
        # Assign remaining qubits randomly.
        for qubit in range(n_servers, n_qubits):
            server = random.randrange(n_servers)
            server_qubits[server] = server_qubits[server] + [qubit]

        # super().__init__(server_coupling, server_qubits)

        server_ebit_mem = {
            server: max((n_qubits-1)//(n_servers), 2)
            for server in server_qubits.keys()
        }

        super().__init__(
            server_coupling=server_coupling,
            server_qubits=server_qubits,
            server_ebit_mem=server_ebit_mem,
        )


class SmallWorldNISQNetwork(NISQNetwork):
    """NISQNetwork with underlying server network that is a small-world
    network. This is to say most servers can be reached from every other
    server by a small number of steps. The ebit memory for each module
    is the the larger of 2 and largest integer less than the average number
    of qubits in each module.
    """

    def __init__(self, n_servers: int, n_qubits: int, **kwargs):
        """Initialisation method for small world network.

        :param n_servers: The number of servers in the network.
        :type n_servers: int
        :param n_qubits: The total number of qubits. Qubits are assigned
            randomly to each server, with at least one per server.
        :type n_qubits: int
        :raises Exception: Raised if the number of qubits is less than the
            number of servers.
        """

        if n_qubits < n_servers:
            raise Exception(
                "The number of qubits must be greater ",
                "than the number of servers."
            )

        k = kwargs.get('k', 2)
        seed = kwargs.get('seed', None)
        p = kwargs.get('p', 0.5)
        tries = kwargs.get('tries', 1000)

        # Generate watts strogatz graph
        graph = nx.connected_watts_strogatz_graph(
            n=n_servers,
            k=k,
            p=p,
            seed=seed,
            tries=tries,
        )
        server_coupling = [list(edge) for edge in graph.edges]

        random.seed(seed)

        # Assign at least one qubit to each server
        server_qubits = {i: [i] for i in range(n_servers)}
        # Assign remaining qubits randomly.
        for qubit in range(n_servers, n_qubits):
            server = random.randrange(n_servers)
            server_qubits[server] = server_qubits[server] + [qubit]

        # super().__init__(server_coupling, server_qubits)

        server_ebit_mem = {
            server: max((n_qubits-1)//(n_servers), 2)
            for server in server_qubits.keys()
        }

        super().__init__(
            server_coupling=server_coupling,
            server_qubits=server_qubits,
            server_ebit_mem=server_ebit_mem,
        )
