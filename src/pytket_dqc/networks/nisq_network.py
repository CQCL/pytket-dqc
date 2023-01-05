from pytket_dqc.networks.server_network import ServerNetwork
from itertools import combinations
import networkx as nx  # type:ignore
from pytket.placement import NoiseAwarePlacement  # type:ignore
from pytket.architecture import Architecture  # type:ignore
from pytket.circuit import Node  # type:ignore
from pytket_dqc.circuits.hypergraph_circuit import HypergraphCircuit
from typing import Tuple, Union
import random


class NISQNetwork(ServerNetwork):
    """Class for the management of NISQ networks of quantum computers. Child
    class of ServerNetwork. Adds additional functionality to manage information
    about the internal architectures of servers.
    """

    def __init__(
        self,
        server_coupling: list[list[int]],
        server_qubits: dict[int, list[int]]
    ) -> None:
        """Initialisation function. Performs checks on inputted network
        description.

        :param server_coupling: List of pairs of server indices. Each pair
            specifies that there is a connection between those two servers.
        :type server_coupling: list[list[int]]
        :param server_qubits: Dictionary from server index to qubits it
            contains.
        :type server_qubits: dict[int, list[int]]
        :raises Exception: Raised if a server is empty.
        :raises Exception: Raised if a server is in more than server.
        """

        super().__init__(server_coupling)

        self.server_qubits = server_qubits

        # Check that each server has a collection of qubits
        # which belong to it specified.
        for server in self.get_server_list():
            if server not in self.server_qubits.keys():
                raise Exception(
                    f"The qubits in server {server}"
                    " have not been specified."
                )

        qubit_list = self.get_qubit_list()
        # Check that each qubit belongs to only one server.
        if not len(qubit_list) == len(set(qubit_list)):
            raise Exception(
                "Qubits may belong to only one server"
                ", and should feature only once per server."
            )

        # Check that the resulting network is connected.
        assert nx.is_connected(self.get_nisq_nx())

    def to_dict(
        self
    ) -> dict[str, Union[list[list[int]], dict[int, list[int]]]]:

        return {
            'server_coupling': self.server_coupling,
            'server_qubits': self.server_qubits,
        }

    def can_implement(self, dist_circ: HypergraphCircuit) -> bool:

        if len(self.get_qubit_list()) < len(dist_circ.get_qubit_vertices()):
            return False
        return True

    def get_qubit_list(self) -> list[int]:
        """Return list of qubit indices.

        :return: List of qubit indices.
        :rtype: list[int]
        """

        # Combine all lists of qubits belonging to each server into one list.
        qubit_list = [
            qubit
            for qubit_list in self.server_qubits.values()
            for qubit in qubit_list
        ]

        return qubit_list

    def get_architecture(self) -> Tuple[Architecture, dict[Node, int]]:
        """Return `tket Architecture
        <https://cqcl.github.io/tket/pytket/api/routing.html#pytket.routing.Architecture>`_  # noqa:E501
        corresponding to network and map from architecture nodes to
        network qubits.

        :return: `tket Architecture
            <https://cqcl.github.io/tket/pytket/api/routing.html#pytket.routing.Architecture>`_  # noqa:E501
            corresponding to network and map from architecture nodes to
            network qubits.
        :rtype: Tuple[Architecture, dict[Node, int]]
        """

        G = self.get_nisq_nx()
        arc = Architecture([(Node(u), Node(v)) for u, v in G.edges])
        # Map from architecture nodes to network qubits
        node_qubit_map = {Node(u): u for u in G.nodes}

        return arc, node_qubit_map

    def get_placer(self) -> Tuple[
        Architecture,
        dict[Node, int],
        NoiseAwarePlacement
    ]:
        """Return `tket NoiseAwarePlacement
        <https://cqcl.github.io/tket/pytket/api/routing.html#pytket.routing.NoiseAwarePlacement>`_  # noqa:E501
        which places onto the network, taking edges between servers to be
        noisy. Return `tket Architecture
        <https://cqcl.github.io/tket/pytket/api/routing.html#pytket.routing.Architecture>`_  # noqa:E501
        corresponding to network and map from architecture nodes to
        network qubits.

        :return: `tket Architecture
            <https://cqcl.github.io/tket/pytket/api/routing.html#pytket.routing.Architecture>`_  # noqa:E501
            corresponding to network. Map from architecture nodes to
            network qubits.
            `tket NoiseAwarePlacement
            <https://cqcl.github.io/tket/pytket/api/routing.html#pytket.routing.NoiseAwarePlacement>`_  # noqa:E501
            which places onto the network, taking edges between servers to be
            noisy.
        :rtype: Tuple[ Architecture, dict[Node, int], NoiseAwarePlacement ]
        """

        G = self.get_nisq_nx()
        link_errors = {}
        # For each edge in network graph, add noise corresponding
        # to edge weight.
        for u, v in G.edges:
            edge_data = G.get_edge_data(u, v)
            link_errors[(Node(u), Node(v))] = edge_data['weight']

        arc, node_qubit_map = self.get_architecture()

        return (
            arc,
            node_qubit_map,
            NoiseAwarePlacement(arc=arc, link_errors=link_errors)
        )

    def get_nisq_nx(self) -> nx.Graph:
        """Return networkx graph corresponding to network.

        :return: networkx graph corresponding to network.
        :rtype: nx.Graph
        """

        G = nx.Graph()

        # For each server, add a connection between each of the qubits
        # internal to the server.
        for qubits in self.server_qubits.values():
            for qubit_connection in combinations(qubits, 2):
                G.add_edge(*qubit_connection, color="red", weight=0)

        # Add edges between one in one server to one qubit in another server
        # if the two servers are connected.
        for u, v in self.server_coupling:
            G.add_edge(
                self.server_qubits[u][0],
                self.server_qubits[v][0],
                color="blue",
                weight=1,
            )

        return G

    def draw_nisq_network(self) -> None:
        """Draw network using netwrokx draw method.
        """

        G = self.get_nisq_nx()
        colors = [G[u][v]["color"] for u, v in G.edges()]
        nx.draw(
            G,
            with_labels=True,
            edge_color=colors,
            pos=nx.nx_agraph.graphviz_layout(G)
        )


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


class AllToAll(NISQNetwork):
    """NISQNetwork consisting of uniformly sized, all to all connected servers.
    """

    def __init__(self, n_servers: int, n_qubits: int):
        """Initialisation function

        :param n_server: Number of servers.
        :type n_server: int
        :param n_qubits: Number of qubits per server
        :type n_qubits: int
        """

        server_coupling = [list(combination) for combination in combinations(
            [i for i in range(n_servers)], 2)]

        qubits = [i for i in range(n_servers*n_qubits)]
        server_qubits_list = [qubits[i:i + n_qubits]
                              for i in range(0, len(qubits), n_qubits)]
        server_qubits = {i: qubits_list for i,
                         qubits_list in enumerate(server_qubits_list)}

        super().__init__(server_coupling, server_qubits)


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
        server_coupling = list(graph.edges)

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
        server_coupling = list(graph.edges)

        random.seed(seed)

        # Assign at least one qubit to each server
        server_qubits = {i: [i] for i in range(n_servers)}
        # Assign remaining qubits randomly.
        for qubit in range(n_servers, n_qubits):
            server = random.randrange(n_servers)
            server_qubits[server] = server_qubits[server] + [qubit]

        super().__init__(server_coupling, server_qubits)
