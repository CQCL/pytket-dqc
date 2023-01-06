from .nisq_network import NISQNetwork
import random
import networkx as nx  # type:ignore


def random_connected_graph(n_nodes: int, edge_prob: float) -> list[list[int]]:
    """Generate random connected graph.

    :param n_nodes: The number of vertices in the graph.
    :type n_nodes: int
    :param edge_prob: The probability of an edge between two vertices.
    :type edge_prob: float
    :raises Exception: Raise if the probability is invalid.
    :return: A coupling map, consisting of a list of pairs of vertices.
    :rtype: list[list[int]]
    """

    if not (edge_prob >= 0 and edge_prob <= 1):
        raise Exception("edge_prob must be between 0 and 1.")

    edge_list = []

    # Build connected graph. Add each node to the graph
    # in sequence so that all are included in the graph.
    for node in range(1, n_nodes):
        edge_list.append({node, random.randrange(node)})

    # Randomly add additional edges with given probability.
    for node in range(n_nodes):
        for connected_node in [i for i in range(n_nodes) if i != node]:
            if random.random() < edge_prob:
                if {node, connected_node} not in edge_list:
                    edge_list.append({node, connected_node})

    G = nx.Graph()
    G.add_edges_from(edge_list)
    assert nx.is_connected(G)

    return [list(edge) for edge in edge_list]


class RandomNISQNetwork(NISQNetwork):
    """NISQNetwok with underlying server network that is random but connected.
    """

    def __init__(self, n_servers: int, n_qubits: int, **kwargs):
        """Initialisation function.

        :param n_servers: The number of servers.
        :type n_servers: int
        :param n_qubits: The total number of qubits.
        :type n_qubits: int
        :param edge_prob: The probability of an edge between two servers,
            defaults to 0.5
        :type edge_prob: float, optional
        :raises Exception: Raised if the number of qubits is less than the
            number of servers.
        """

        if n_qubits < n_servers:
            raise Exception(
                "The number of qubits must be greater ",
                "than the number of servers.")

        edge_prob = kwargs.get("edge_prob", 1/(n_servers-1))
        seed = kwargs.get("seed", None)

        random.seed(seed)

        server_coupling = random_connected_graph(n_servers, edge_prob)

        # Assign at least one qubit to each server
        server_qubits = {i: [i] for i in range(n_servers)}
        # Assign remaining qubits randomly.
        for qubit in range(n_servers, n_qubits):
            server = random.randrange(n_servers)
            server_qubits[server] = server_qubits[server] + [qubit]

        super().__init__(server_coupling, server_qubits)


class ScaleFreeNISQNetwork(NISQNetwork):
    """NISQNetwork with underlying server network that is scale-free. This is
    to say one whose degree distribution follows a power law.
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

        m = kwargs.get('m', 2)
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

        super().__init__(server_coupling, server_qubits)


class SmallWorldNISQNetwork(NISQNetwork):
    """NISQNetwork with underlying server network that is a small-world
    network. This is to say most servers can be reached from every other
    server by a small number of steps
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

        k = kwargs.get('k', 4)
        seed = kwargs.get('seed', None)
        p = kwargs.get('p', 0.5)
        tries = kwargs.get('tries', 1000)

        # Generate barabasi albert graph
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

        super().__init__(server_coupling, server_qubits)
