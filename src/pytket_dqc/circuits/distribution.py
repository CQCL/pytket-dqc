from pytket_dqc.circuits import HypergraphCircuit, Hyperedge
from pytket_dqc.placement import Placement
from pytket_dqc.networks import NISQNetwork
from pytket_dqc.utils import steiner_tree, check_equivalence
from pytket_dqc.utils.gateset import start_proc, end_proc
from pytket_dqc.utils.circuit_analysis import all_cu1_local, _cost_from_circuit
from pytket import Circuit, OpType, Qubit
import networkx as nx  # type: ignore
from numpy import isclose  # type: ignore
from typing import NamedTuple


class Distribution:
    """Class containing all information required to generate pytket circuit.
    """

    def __init__(
        self,
        circuit: HypergraphCircuit,
        placement: Placement,
        network: NISQNetwork,
    ):
        """Initialisation function for Distribution

        :param circuit: Circuit to be distributed, including its hypergraph.
        :type circuit: HypergraphCircuit
        :param placement: A placement of the qubits and gates onto servers.
        :type placement: Placement
        :param network: Network onto which circuit is distributed.
        :type network: NISQNetwork
        :raises Exception: Raised if the placement is not valid for the
            circuit.
        :raises Exception: Raised if the placement is not valid for the
            packets.
        """

        self.circuit = circuit
        self.placement = placement
        self.network = network

    def is_valid(self) -> bool:
        """Check that this distribution can be implemented.
        """

        # TODO: There may be some other checks that we want to do here to check
        # that the phypergraph is not totally nonsensical.
        return self.placement.is_valid(self.circuit, self.network)

    def cost(self) -> int:
        """Return the number of ebits required for this distribution.
        """
        if not self.is_valid():
            raise Exception("This is not a valid distribution")

        cost = 0
        for hyperedge in self.circuit.hyperedge_list:
            cost += self.hyperedge_cost(hyperedge)
        return cost

    def hyperedge_cost(self, hyperedge: Hyperedge, **kwargs) -> int:
        """First, we check whether the hyperedge requires H-embeddings to be
        implemented. If not, we calculate its cost by counting the number of
        edges in the Steiner tree connecting all required servers. Otherwise,
        the cost of implementing the hyperedge is calculated using an "as lazy
        as possible" (ALAP) algorithm which we expect not to be optimal, but
        decent enough. In the latter case, both reduction of ebit cost via
        Steiner trees and embedding are considered.

        Note: the cost only takes into account the ebits required to
        distribute the gates in the hyperedge; it does not consider the ebit
        cost of distributing the embedded gates, since that will be calculated
        by calling ``hyperedge_cost`` that the embedded gates belong to.
        However, we do guarantee that the correction gates added during the
        embedding will not require extra ebits when distributed.

        :param hyperedge: The hyperedge whose cost is to be calculated.
        :type hyperedge: Hyperedge

        :key server_tree: The connectivity tree (subgraph of ``server_graph``)
            that should be used. If not provided, this function will find a
            Steiner tree.
        :type server_tree: nx.Graph
        :key server_graph: The network's ``server_graph``. If not provided,
            this is calculated from ``self.network``. Meant to save a call.
        :type server_graph: nx.Graph
        :key h_embedding: Indicates whether or not the hyperedge requires
            an H-embedding to be implemented. If not provided, it is checked.
        :type h_embedding: bool

        :return: The cost of the hyperedge.
        :rtype: int
        """
        tree = kwargs.get("server_tree", None)
        server_graph = kwargs.get("server_graph", None)
        h_embedding = kwargs.get("h_embedding", None)

        if hyperedge.weight != 1:
            raise Exception(
                "Hyperedges with weight other than 1 \
                 are not currently supported"
            )

        dist_circ = self.circuit
        placement_map = self.placement.placement

        # Extract hyperedge data
        shared_qubit = dist_circ.get_qubit_vertex(hyperedge)
        home_server = placement_map[shared_qubit]
        servers = [placement_map[v] for v in hyperedge.vertices]
        # Obtain the server graph if it is not given
        if server_graph is None:
            server_graph = self.network.get_server_nx()
        # Obtain the Steiner tree or check that the one given is valid
        if tree is None:
            tree = steiner_tree(server_graph, servers)
        else:
            assert all(s in tree.nodes for s in servers)

        # If not known, check if H-embedding is required for this hyperedge
        if h_embedding is None:
            h_embedding = dist_circ.h_embedding_required(hyperedge)

        # If H-embedding is not required, we can easily calculate the cost
        if not h_embedding:
            return len(tree.edges)

        # Otherwise, we need to run ALAP
        else:
            # Collect all of the commands between the first and last gates in
            # the hyperedge. Ignore those do not act on the shared qubit.
            commands = dist_circ.get_hyperedge_subcircuit(hyperedge)

            # We will use the fact that, by construction, the index of the
            # vertices is ordered (qubits first, then gates left to right)
            vertices = sorted(hyperedge.vertices)
            assert vertices.pop(0) == shared_qubit

            cost = 0
            currently_h_embedding = False  # Switched when finding a Hadamard
            connected_servers = {home_server}  # With access to shared_qubit
            for command in commands:

                if command.op.type == OpType.H:
                    currently_h_embedding = not currently_h_embedding

                elif command.op.type in [OpType.X, OpType.Z]:
                    pass  # These gates can always be embedded

                elif command.op.type == OpType.Rz:
                    assert (
                        not currently_h_embedding
                        # Gate has a phase multiple of pi (i.e. I or Z)
                        or isclose(command.op.params[0] % 1, 0)
                        or isclose(command.op.params[0] % 1, 1)
                    )

                elif command.op.type == OpType.CU1:

                    if currently_h_embedding:  # Gate to be embedded
                        assert isclose(command.op.params[0] % 2, 1)  # CZ gate

                        q_vertices = [
                            dist_circ.get_vertex_of_qubit(q)
                            for q in command.qubits
                        ]
                        remote_vertex = [
                            q for q in q_vertices if q != shared_qubit
                        ][0]
                        remote_server = placement_map[remote_vertex]

                        # Only servers in the shortest path from remote_server
                        # to home_server are left intact. The rest of the
                        # servers need to be disconnected since, otherwise,
                        # extra ebits would be required to implement the new
                        # correction gates that would be introduced.
                        #
                        # Note: the shortest path is found in server_graph
                        # instead of in the tree since that is how the
                        # embedded hyperedge would be implemented if it were
                        # not embedded (it only connects two servers, so its
                        # Steiner tree is the shortest path). If we used
                        # some other path then we would be changing the
                        # way the embedded gates are distributed, possibly
                        # increasing the cost of their distribution, hence,
                        # not following Junyi's main design principle.
                        #
                        # Note: by the conditions of embeddability, all gates
                        # that are being embedded simultaneously act on the
                        # same two servers.
                        connection_path = nx.shortest_path(
                            server_graph, home_server, remote_server
                        )
                        connected_servers = connected_servers.intersection(
                            connection_path
                        )

                    else:  # Gate to be distributed (or already local)

                        # Get the server where the gate is to be implemented
                        gate_vertex = vertices.pop(0)
                        assert (
                            dist_circ._vertex_circuit_map[gate_vertex][
                                "command"
                            ]
                            == command
                        )
                        gate_server = placement_map[gate_vertex]
                        # If gate_server doesn't have access to shared_qubit
                        # update the cost, adding the necessary ebits
                        if gate_server not in connected_servers:
                            connection_path = set(
                                nx.shortest_path(
                                    tree, home_server, gate_server
                                )
                            )
                            required_connections = connection_path.difference(
                                connected_servers
                            )
                            connected_servers.update(required_connections)
                            cost += len(required_connections)
            return cost

    def get_qubit_mapping(self) -> dict[Qubit, Qubit]:
        """Mapping from circuit (logical) qubits to server (hardware) qubits.
        """
        if not self.is_valid():
            raise Exception("The distribution of the circuit is not valid!")

        qubit_map = {}
        current_in_server = {s: 0 for s in self.network.get_server_list()}
        for v in self.circuit.get_qubit_vertices():
            server = self.placement.placement[v]
            circ_qubit = self.circuit.get_qubit_of_vertex(v)
            hw_qubit = Qubit(f"Server {server}", current_in_server[server])
            current_in_server[server] += 1
            qubit_map[circ_qubit] = hw_qubit

        return qubit_map

    def to_pytket_circuit(self) -> Circuit:
        if not self.is_valid():
            raise Exception("The distribution of the circuit is not valid!")

        # -- SCOPE VARIABLES -- #
        # Accessible to the internal class below
        hyp_circ = self.circuit
        server_graph = self.network.get_server_nx()
        placement_map = self.placement.placement
        qubit_mapping = self.get_qubit_mapping()

        # -- INTERNAL CLASSES -- #

        class EjppAction(NamedTuple):
            """Encodes the information to create Starting and EndingProcesses
            """

            from_qubit: Qubit
            to_qubit: Qubit

        class LinkManager:
            """An internal class dedicated to managing the hardware qubits
            that store the ebits (i.e. link qubits).
            """

            def __init__(self, servers: list[int]):
                # A dictionary of serverId to list of available link qubits.
                self.available: dict[int, list[Qubit]] = {
                    s: [] for s in servers
                }
                # A dictionary of serverId to list of occupied link qubits.
                self.occupied: dict[int, list[Qubit]] = {
                    s: [] for s in servers
                }
                # A dictionary matching a (qubit, server) pair with the qubit
                # in ``server`` that has a "copy" of the data in ``qubit``,
                # where ``qubit`` is a Qubit in the original circuit.
                self.link_qubit_dict: dict[tuple[Qubit, int], Qubit] = {}

            def request_link_qubit(self, server: int) -> Qubit:
                """Returns an available link qubit in ``server``. If there are
                none, it creates a new one.
                """
                if not self.available[server]:
                    next_id = len(self.occupied[server])
                    qubit = Qubit(f"Server {server} Link Register", next_id)
                else:
                    qubit = self.available[server].pop()

                self.occupied[server].append(qubit)
                return qubit

            def release_link_qubit(self, link_qubit: Qubit):
                """Releases ``link_qubit``, making it again available.
                """

                # Singleton list of servers containing ``link_qubit``
                servers = [
                    s for s, qs in self.occupied.items() if link_qubit in qs
                ]
                assert len(servers) == 1
                its_server = servers[0]

                self.occupied[its_server].remove(link_qubit)
                self.available[its_server].append(link_qubit)

            def connected_servers(self, circ_qubit: Qubit) -> list[int]:
                """Return the list of servers currently holding a copy
                of ``circ_qubit``. Does not include its home server.
                """
                return [
                    s
                    for q, s in self.link_qubit_dict.keys()
                    if q == circ_qubit
                ]

            def get_link_qubit(self, circ_qubit: Qubit, server: int) -> Qubit:
                """If ``server`` is the home server of ``circ_qubit``, the HW
                qubit corresponding to ``circ_qubit`` is returned. Otherwise,
                we query ``link_qubit_dict`` to retrieve the appropriate link
                qubit.
                """
                q_vertex = hyp_circ.get_vertex_of_qubit(circ_qubit)
                if placement_map[q_vertex] == server:
                    return qubit_mapping[circ_qubit]
                else:
                    return self.link_qubit_dict[(circ_qubit, server)]

            def start_link(
                self, hyperedge: Hyperedge, target: int
            ) -> list[EjppAction]:
                """Find sequence of StartingProcesses required to share the
                qubit of ``hyperedge`` with ``target`` server.
                """
                # NOTE: If all of the ``hyp_servers`` are to be connected,
                # then the routine below will "grow" the tree branch by
                # branch. And, since EJPP starting processes commute with each
                # other, the ordering of how it is "grown" does not matter.

                # Extract hyperedge data
                q_vertex = hyp_circ.get_qubit_vertex(hyperedge)
                circ_qubit = hyp_circ.get_qubit_of_vertex(q_vertex)
                home_server = placement_map[q_vertex]
                hyp_servers = [placement_map[v] for v in hyperedge.vertices]
                tree = steiner_tree(server_graph, hyp_servers)
                assert target in hyp_servers

                # For each server connected to ``circ_qubit``, find the
                # shortest path to the target server and use the one that
                # is shortest among them
                connected_servers = self.connected_servers(circ_qubit)
                connected_servers.append(home_server)
                best_path = None
                for c_server in connected_servers:
                    connection_path = nx.shortest_path(tree, c_server, target)
                    if best_path is None or len(connection_path) < len(
                        best_path
                    ):
                        best_path = connection_path
                assert best_path is not None
                # NOTE: the ``best_path`` will only contain
                # one server from ``connected_servers``. Proof: if
                # it had two, the shortest path from the second
                # would be shorter => contradiction

                # Go over ``best_path`` step by step adding the required
                # starting EjppActions
                starting_actions = []
                source = best_path.pop(0)
                last_link_qubit = self.get_link_qubit(circ_qubit, source)
                for next_server in best_path:
                    # Retrieve an available link qubit to populate
                    this_link_qubit = self.request_link_qubit(next_server)
                    # Add the EjppAction to create the connection
                    starting_actions.append(
                        EjppAction(
                            from_qubit=last_link_qubit,
                            to_qubit=this_link_qubit,
                        )
                    )
                    # Add the link qubit to the dictionary
                    self.link_qubit_dict[
                        (circ_qubit, next_server)
                    ] = this_link_qubit
                    last_link_qubit = this_link_qubit

                return starting_actions

            def end_links(
                self, circ_qubit: Qubit, targets: list[int]
            ) -> list[EjppAction]:
                """Find the sequence of EndingProcesses required to end the
                connection of ``circ_qubit`` to each server in ``targets``.

                NOTE: ``circ_qubit`` is a qubit in the original circuit,
                i.e. a key in ``qubit_mapping``.
                """
                assert circ_qubit in qubit_mapping.keys()

                # Find the HW qubit holding ``circ_qubit``
                home_link = qubit_mapping[circ_qubit]

                # Disconnect each server in ``targets``
                ending_actions = []
                for target in targets:
                    # Find the corresponding HW qubit
                    target_link = self.get_link_qubit(circ_qubit, target)
                    # Disconnect ``target_link``
                    ending_actions.append(
                        EjppAction(from_qubit=target_link, to_qubit=home_link,)
                    )
                    # Release the HW qubit acting as ``target_link``
                    self.release_link_qubit(target_link)
                    # Delete its entry from the dictionary
                    del self.link_qubit_dict[(circ_qubit, target)]

                return ending_actions

        # -- CIRCUIT GENERATION -- #
        new_circ = Circuit()
        # Add the qubits to the circuit
        for hw_qubit in qubit_mapping.values():
            new_circ.add_qubit(hw_qubit)

        # Read the original circuit from left to right and, as we go:
        #   (1) add the required EJPP processes
        #   (2) distribute nonlocal gates
        #   (3) add the correction gates required for embedding
        # Some data to keep around while iterating over the commands:
        #
        # Map from qubits (from the original circuit) to a boolean flag
        # indicating whether we currently are within an H-embedding unit
        currently_h_embedding = {q: False for q in qubit_mapping.keys()}
        # The keys of the following map contain qubits (from the original
        # circuit) currently with an H-embedding unit acting on it
        # The values will contain the corresponding link qubit (on the
        # remote server) that is kept alive thanks to the embedding
        embedding: dict[Qubit, Qubit] = {}
        # The LinkManager that will deal with the link qubits
        linkman = LinkManager(self.network.get_server_list())

        # Iterate over the commands of the original circuit
        commands = hyp_circ._circuit.get_commands()
        orig_cu1_count = 0
        for cmd_idx, cmd in enumerate(commands):

            if cmd.op.type == OpType.H:
                q = cmd.qubits[0]
                # Append the gate to ``new_circ``
                new_circ.H(qubit_mapping[q])
                # The presence of an H gate indicates the beginning or end
                # of an H-embedding on the qubit
                #
                # Whether this H is the beginning or end of an H-embedding
                # unit, there are two cases to consider:
                #
                # (Case 1) There is at least one CU1 gate (actually, a CZ)
                # within the H-embedding unit.
                # (Case 2) There is no CU1 gate within the H-embedding unit.
                #
                if not currently_h_embedding[q]:  # Start an embedding unit
                    currently_h_embedding[q] = True
                    # Look through the rest of the circuit and figure out
                    # if we are in case 1 or case 2
                    found_CU1_gate = None
                    for gate in commands[(cmd_idx + 1):]:
                        if gate.op.type == OpType.CU1 and q in gate.qubits:
                            found_CU1_gate = gate
                            break
                        # Otherwise, stop when finding an H gate on q
                        elif gate.op.type == OpType.H and q in gate.qubits:
                            break

                    if found_CU1_gate is not None:  # Case 1
                        # All connections to servers must be closed, except
                        # that of the remote server the CU1 gate acts on
                        #
                        # NOTE: According to the conditions of embeddability,
                        # the embedded CU1 gates all act on the same servers
                        remote_qubit = [
                            rq for rq in found_CU1_gate.qubits if rq != q
                        ][0]
                        remote_vertex = hyp_circ.get_vertex_of_qubit(
                            remote_qubit
                        )
                        remote_server = placement_map[remote_vertex]
                        servers = [
                            s
                            for s in linkman.connected_servers(q)
                        ]
                        assert remote_server in servers
                        servers.remove(remote_server)

                        # Close the connections
                        for ejpp_end in linkman.end_links(q, servers):
                            new_circ.add_custom_gate(
                                end_proc,
                                [],
                                [ejpp_end.from_qubit, ejpp_end.to_qubit],
                            )

                        # Retrieve the link on the remote server
                        remote_link = linkman.get_link_qubit(q, remote_server)
                        # Remove it from the dictionary so that it is not
                        # mistaken with a link qubit used to distribute the
                        # embedded gates
                        del linkman.link_qubit_dict[(q, remote_server)]
                        # Save the link qubit on the embedding dictionary for
                        # later reference
                        embedding[q] = remote_link
                        # Append the correction Hadamard to the circuit
                        new_circ.H(remote_link)

                    else:  # Case 2
                        # We do not need to close connections but we
                        # must place an H gate on each link qubit
                        for server in linkman.connected_servers(q):
                            # Append the correction Hadamard to the circuit
                            new_circ.H(linkman.get_link_qubit(q, server))
                            # NOTE: The ``embedding`` map is not updated.
                            # This fact will be used when finding the H gate
                            # that ends this embedding unit, as the absence
                            # of the key will indicate we did not close
                            # the rest of the connections (Case 2)

                # Otherwise, this Hadamard is the end of the H-embedding unit
                else:
                    currently_h_embedding[q] = False

                    if q in embedding.keys():  # Case 1
                        # Retrieve the remote link that survived the embedding
                        remote_link = embedding[q]
                        remote_vertex = hyp_circ.get_vertex_of_qubit(
                            remote_qubit
                        )
                        rmt_server = placement_map[remote_vertex]
                        # Restore the dictionaries
                        del embedding[q]
                        linkman.link_qubit_dict[(q, rmt_server)] = remote_link

                        # Append the correction Hadamard to the circuit
                        new_circ.H(remote_link)

                    else:  # Case 2
                        # Append a correction Hadamard to each link qubit
                        for server in linkman.connected_servers(q):
                            new_circ.H(linkman.get_link_qubit(q, server))

            elif cmd.op.type == OpType.Rz:
                q = cmd.qubits[0]
                phase = cmd.op.params[0]
                # Append the gate to ``new_circ``
                new_circ.Rz(phase, qubit_mapping[q])

                # If not H-embedding, nothing needs to be done; otherwise:
                if currently_h_embedding[q]:
                    # The phase must be multiple of pi (either I or Z gate)
                    assert isclose(phase % 1, 0) or isclose(phase % 1, 1)
                    # If Z gate, apply it to all connected servers
                    if isclose(phase % 2, 1):  # Z gate
                        c_servers = linkman.connected_servers(q)
                        for server in c_servers:
                            link_qubit = linkman.link_qubit_dict[(q, server)]
                            new_circ.Z(link_qubit)
                    # Otherwise, it is an identity gate and we do nothing

            elif cmd.op.type == OpType.Z:
                q = cmd.qubits[0]
                # Append the gate to ``new_circ``
                new_circ.Z(qubit_mapping[q])

                # If not H-embedding, nothing needs to be done; otherwise:
                if currently_h_embedding[q]:
                    # Apply the gate to all connected servers
                    c_servers = linkman.connected_servers(q)
                    for server in c_servers:
                        link_qubit = linkman.link_qubit_dict[(q, server)]
                        new_circ.Z(link_qubit)

            elif cmd.op.type == OpType.X:
                q = cmd.qubits[0]
                # Append the gate to ``new_circ``
                new_circ.X(qubit_mapping[q])

                # If H-embedding, nothing needs to be done; otherwise:
                if not currently_h_embedding[q]:
                    # Apply the gate to all connected servers
                    c_servers = linkman.connected_servers(q)
                    for server in c_servers:
                        link_qubit = linkman.link_qubit_dict[(q, server)]
                        new_circ.X(link_qubit)

            elif cmd.op.type == OpType.CU1:
                phase = cmd.op.params[0]
                q0 = cmd.qubits[0]
                q1 = cmd.qubits[1]
                # Find the vertices of these
                v_q0 = hyp_circ.get_vertex_of_qubit(q0)
                v_q1 = hyp_circ.get_vertex_of_qubit(q1)
                v_gate = orig_cu1_count + len(qubit_mapping)
                # Find the server where the gate should be implemented
                target_server = placement_map[v_gate]
                # Find the hyperedges of these
                hyps0 = hyp_circ.get_hyperedges_containing([v_gate, v_q0])
                assert len(hyps0) == 1
                hyp0 = hyps0[0]
                hyps1 = hyp_circ.get_hyperedges_containing([v_gate, v_q1])
                assert len(hyps1) == 1
                hyp1 = hyps1[0]

                # Add the required starting processes (if any)
                start_actions = linkman.start_link(
                    hyp0, target_server
                ) + linkman.start_link(hyp1, target_server)
                for ejpp_start in start_actions:
                    if ejpp_start.to_qubit not in new_circ.qubits:
                        new_circ.add_qubit(ejpp_start.to_qubit)
                    new_circ.add_custom_gate(
                        start_proc,
                        [],
                        [ejpp_start.from_qubit, ejpp_start.to_qubit],
                    )

                # Append the gate to the circuit
                new_circ.add_gate(
                    OpType.CU1, phase, [qubit_mapping[q0], qubit_mapping[q1]]
                )
                # Append correction gates if within an H-embedding unit
                if currently_h_embedding[q0]:
                    link_qubit = embedding[q0]
                    new_circ.add_gate(
                        OpType.CU1, phase, [link_qubit, qubit_mapping[q1]]
                    )
                if currently_h_embedding[q1]:
                    link_qubit = embedding[q1]
                    new_circ.add_gate(
                        OpType.CU1, phase, [qubit_mapping[q0], link_qubit]
                    )

                # Count increases by one; we only count CU1 in the original
                # circuit because we use this to derive the gate vertex id
                orig_cu1_count += 1

        # Sanity check: this is guaranteed by the fact that an H-embedding
        # unit is always defined between two CU1 gates, so it cannot possibly
        # be that at the end of the circuit an H-embedding unit has not yet
        # reached its ending Hadamard.
        assert all(not currently_h_embedding[q] for q in qubit_mapping.keys())
        assert not embedding
        # Finally, close all remaining connections
        for q in qubit_mapping.keys():
            src_qubit = qubit_mapping[q]
            for c_server in linkman.connected_servers(q):
                link_qubit = linkman.link_qubit_dict[(q, c_server)]
                new_circ.add_custom_gate(end_proc, [], [link_qubit, src_qubit])

        # Final sanity checks
        assert all_cu1_local(new_circ)
        assert check_equivalence(
            self.circuit.get_circuit(), new_circ, qubit_mapping
        )
        assert _cost_from_circuit(new_circ) == self.cost()

        return new_circ
