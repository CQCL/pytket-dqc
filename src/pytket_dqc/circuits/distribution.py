# Copyright 2023 Quantinuum and The University of Tokyo
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations
from pytket_dqc.circuits import HypergraphCircuit, Hyperedge
from pytket_dqc.placement import Placement
from pytket_dqc.networks import NISQNetwork
from pytket_dqc.utils import steiner_tree, check_equivalence, ConstraintException
from pytket_dqc.utils.gateset import (
    start_proc,
    is_start_proc,
    end_proc,
    is_end_proc,
    origin_of_start_proc,
)
from pytket_dqc.utils.circuit_analysis import (
    all_cu1_local,
    ebit_cost,
    get_server_id,
    is_link_qubit,
)
from pytket import Circuit, OpType, Qubit
from pytket.circuit import Command
import networkx as nx  # type: ignore
from numpy import isclose
import warnings
from typing import NamedTuple
from .hypergraph import Vertex


class Distribution:
    """Class containing all information required to generate pytket circuit."""

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

    def __eq__(self, other) -> bool:
        """Check equality based on equality of components"""
        if isinstance(other, Distribution):
            return (
                self.circuit == other.circuit
                and self.placement == other.placement
                and self.network == other.network
            )
        return False

    def to_dict(self) -> dict[str, dict]:
        """Generate JSON serialisable dictionary representation of
        the Distribution.

        :return: JSON serialisable dictionary representation of
            the Distribution.
        :rtype: dict[str, dict]
        """
        return {
            "circuit": self.circuit.to_dict(),
            "placement": self.placement.to_dict(),
            "network": self.network.to_dict(),
        }

    @classmethod
    def from_dict(cls, distribution_dict: dict[str, dict]) -> Distribution:
        """Construct ``Distribution`` instance from JSON serialisable
        dictionary representation of the Distribution.

        :param distribution_dict: JSON serialisable dictionary
            representation of the Distribution
        :type distribution_dict: dict[str, dict]
        :return: Distribution instance constructed from distribution_dict.
        :rtype: Distribution
        """
        return cls(
            circuit=HypergraphCircuit.from_dict(distribution_dict["circuit"]),
            placement=Placement.from_dict(distribution_dict["placement"]),
            network=NISQNetwork.from_dict(distribution_dict["network"]),
        )

    def is_valid(self) -> bool:
        """Check that this distribution can be implemented."""

        # TODO: There may be some other checks that we want to do here to check
        # that the hypergraph is not totally nonsensical.
        return self.placement.is_valid(self.circuit, self.network)

    def cost(self) -> int:
        """Return the number of ebits required to implement this distribution.

        NOTE: this function does not guarantee that the distribution being
        analysed satisfies the bound to the link qubit registers. If you wish
        to take the bound into account, call ``to_pytket_circuit`` before
        calling this function. Alternatively, call ``ebit_cost`` on the
        distributed circuit.

        :return: The number of ebits used in this distribution.
        :rtype: int
        """
        if not self.is_valid():
            raise Exception("This is not a valid distribution")

        cost = 0
        for hyperedge in self.circuit.hyperedge_list:
            cost += self.hyperedge_cost(hyperedge)
        return cost

    def non_local_gate_count(self) -> int:
        """Scan the distribution and return the number of non-local gates.
        A non-local gate is a 2-qubit gate that acts in a server not
        containing one of the two qubits it acts on.

        :return: A list of non-local gates
        :rtype: int
        """

        return len(self.non_local_gate_list())

    def non_local_gate_list(self) -> list[Vertex]:
        """Scan the distribution and return a list of non-local gates.
        A non-local gate is a 2-qubit gate that acts in a server not
        containing one of the two qubits it acts on.

        :return: The number of non-local gates
        :rtype: list[Vertex]
        """

        non_local_list = []

        for vertex in self.circuit.vertex_list:
            if not self.circuit.is_qubit_vertex(vertex):
                # Qubits gate acts on in original circuit
                q_1, q_2 = self.circuit.get_gate_of_vertex(vertex).qubits

                # Vertices of these qubits
                v_1 = self.circuit.get_vertex_of_qubit(q_1)
                v_2 = self.circuit.get_vertex_of_qubit(q_2)

                # Servers to which the qubits have been assigned
                s_1 = self.placement.placement[v_1]
                s_2 = self.placement.placement[v_2]

                # Server to which the gate has been assigned
                s_g = self.placement.placement[vertex]

                # Append if non-local
                if not ((s_1 == s_g) and (s_2 == s_g)):
                    non_local_list.append(vertex)

        return non_local_list

    def detached_gate_list(self) -> list[Vertex]:
        """Scan the distribution and return a list of detached gates in it.
        A detached gate is a 2-qubit gate that acts on link qubits on both
        ends; i.e. it is implemented away from both of its home servers.

        :return: A list of detached gates
        :rtype: list[Vertex]
        """

        detached_list = []

        for vertex in self.circuit.vertex_list:
            if not self.circuit.is_qubit_vertex(vertex):
                # Qubits gate acts on in original circuit
                q_1, q_2 = self.circuit.get_gate_of_vertex(vertex).qubits

                # Vertices of these qubits
                v_1 = self.circuit.get_vertex_of_qubit(q_1)
                v_2 = self.circuit.get_vertex_of_qubit(q_2)

                # Servers to which the qubits have been assigned
                s_1 = self.placement.placement[v_1]
                s_2 = self.placement.placement[v_2]

                # Server to which the gate has been assigned
                s_g = self.placement.placement[vertex]

                # Append if detached
                if not ((s_1 == s_g) or (s_2 == s_g)):
                    detached_list.append(vertex)

        return detached_list

    def detached_gate_count(self) -> int:
        """Scan the distribution and return the number of detached gates in it.
        A detached gate is a 2-qubit gate that acts on link qubits on both
        ends; i.e. it is implemented away from both of its home servers.

        :return: The number of detached gates
        :rtype: int
        """

        return len(self.detached_gate_list())

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

        :key server_tree: The connectivity tree that should be used.
            If not provided, this function will find a Steiner tree.
        :type server_tree: nx.Graph
        :key requires_h_embedded_cu1: If not provided, it is checked.
        :type requires_h_embedded_cu1: bool

        :return: The cost of the hyperedge.
        :rtype: int
        """

        tree = kwargs.get("server_tree", None)
        requires_h_embedded_cu1 = kwargs.get("requires_h_embedded_cu1", None)

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
        # Obtain the Steiner tree or check that the one given is valid
        if tree is None:
            tree = steiner_tree(self.network.get_server_nx(), servers)
        else:
            assert all(s in tree.nodes for s in servers)

        # If not known, check if H-embedding of CU1 is required
        if requires_h_embedded_cu1 is None:
            requires_h_embedded_cu1 = dist_circ.requires_h_embedded_cu1(hyperedge)

        # If not required, we can easily calculate the cost
        if not requires_h_embedded_cu1:
            return len(tree.edges)

        # Otherwise, we need to run ALAP
        else:
            # Collect all of the commands between the first and last gates in
            # the hyperedge. Ignore those that do not act on the shared qubit.
            # The gates in between embedded CU1 gates are guaranteed to be of
            # the form [Rz, H, Rz, H]; i.e. an explicit Euler decomposition.
            # Furthermore, the Rz gates at either side of an embedded CU1 gate
            # are squashed together (that's why there's no Rz at the end of
            # the Euler decomposition).
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

                elif command.op.type == OpType.Rz:
                    assert (
                        not currently_h_embedding
                        # Gate has a phase multiple of pi (i.e. I or Z)
                        or isclose(command.op.params[0] % 1, 0)
                        or isclose(command.op.params[0] % 1, 1)
                    )

                elif command.op.type == OpType.CU1:
                    if currently_h_embedding:  # The gate is to be H-embedded
                        assert isclose(command.op.params[0] % 2, 1)  # CZ gate

                        q_vertices = [
                            dist_circ.get_vertex_of_qubit(q) for q in command.qubits
                        ]
                        remote_vertex = [q for q in q_vertices if q != shared_qubit][0]
                        remote_server = placement_map[remote_vertex]

                        # The only two servers from ``connected_servers`` that
                        # are left intact are the ``home_server`` and the
                        # ``remote_server``. All other servers must be
                        # disconnected.
                        #
                        # NOTE: Done for the sake of simplicity. If we don't,
                        # certain correction gates wouldn't be free or be free
                        # only if the path of the EJPP process distributing
                        # the embedded gate could be reused to implement the
                        # correction gates. This is not trivial and not
                        # taken into account on Junyi's approach since it
                        # assumes only two servers / all-to-all connectivity
                        #
                        # NOTE: by the condition of H-embeddability, all gates
                        # that are being embedded simultaneously act on the
                        # same two distinct servers.
                        connected_servers = connected_servers.intersection(
                            {home_server, remote_server}
                        )
                        assert home_server != remote_server

                    # If the command does not match the vertex, then this
                    # CU1 gate is meant to be D-embedded
                    elif command != dist_circ.get_gate_of_vertex(vertices[0]):
                        pass  # Nothing needs to be done
                    else:  # Gate to be distributed (or already local)
                        # Get the server where the gate is to be implemented
                        gate_vertex = vertices.pop(0)
                        gate_server = placement_map[gate_vertex]
                        # If gate_server doesn't have access to shared_qubit
                        # update the cost, adding the necessary ebits
                        if gate_server not in connected_servers:
                            # For each server in ``connected_servers`` find
                            # the shortest path to ``gate_server`` and use
                            # the one that is shortest among them
                            best_path = None
                            for c_server in connected_servers:
                                connection_path = nx.shortest_path(
                                    tree, c_server, gate_server
                                )
                                # fmt: off
                                if (
                                    best_path is None
                                    or len(connection_path) < len(best_path)
                                ):
                                    best_path = connection_path
                                # fmt: on
                            assert best_path is not None
                            # The first element of the path is a ``c_server``
                            # so the actual cost is the length minus one
                            #
                            # NOTE: the ``best_path`` will only contain
                            # one server from ``connected_servers``. If
                            # it had two, the shortest path from the second
                            # would be shorter => contradiction
                            connected_servers.update(best_path)
                            cost += len(best_path) - 1
            # Sanity check: all gate vertices have been considered
            assert not vertices
            return cost

    def get_qubit_mapping(self) -> dict[Qubit, Qubit]:
        """Mapping from the names of the original circuit's qubits to the
        names of the hardware qubits, including whether they are link qubits
        or computation qubits and the module they belong to.

        :return: The mapping of circuit qubits to hardware qubits.
        :rtype: dict[Qubit, Qubit]
        """
        if not self.is_valid():
            raise Exception("The distribution of the circuit is not valid!")

        qubit_map = {}
        current_in_server = {s: 0 for s in self.network.get_server_list()}
        for v in self.circuit.get_qubit_vertices():
            server = self.placement.placement[v]
            circ_qubit = self.circuit.get_qubit_of_vertex(v)
            hw_qubit = Qubit(f"server_{server}", current_in_server[server])
            current_in_server[server] += 1
            qubit_map[circ_qubit] = hw_qubit

        return qubit_map

    def to_pytket_circuit(
        self, satisfy_bound: bool = True, allow_update: bool = False
    ) -> Circuit:
        """Generate the circuit corresponding to this `Distribution`.

        :param satisfy_bound: Whether the output circuit is only allowed
            to use as many link qubits per module as establisehd by the
            network's communication capacity (if provided). Optional
            parameter, defaults to True.
        :param allow_update: Whether the `Distribution` may be altered by
            this method, in order to make it satisfy the network's
            communication capacity (in case it has been bounded). Optional
            parameter, defaults to False.
        :raise ``ConstraintException``: If a server's communication capacity
            is exceeded, `satisfy_bound` was set to True and `allow_update`
            was set to False.
        """
        if not self.is_valid():
            raise Exception("The distribution of the circuit is not valid!")

        # -- SCOPE VARIABLES -- #
        # Accessible to the internal class below
        hyp_circ = self.circuit
        server_graph = self.network.get_server_nx()
        placement_map = self.placement.placement
        qubit_mapping = self.get_qubit_mapping()
        server_ebit_mem = self.network.server_ebit_mem

        # -- INTERNAL CLASSES -- #
        class EjppAction(NamedTuple):
            """Encodes the information to create Starting and EndingProcesses"""

            from_qubit: Qubit
            to_qubit: Qubit

        class LinkManager:
            """An internal class dedicated to managing the hardware qubits
            that store the ebits (i.e. link qubits).
            This class has been designed for it to be used within
            `to_pytket_circuit_one_hyperedge` and a new instance of
            `LinkManager` should be created per call to it.
            The key methods LinkManager provides are:

            `start_link` which creates a link qubit and returns
            the starting EJPP processes required to entangle it,
            `end_links` which releases link qubits and returns the
            corresponding ending EJPP processes,
            `update_occupation` to be used when encountering an EJPP
            process while reading a circuit, so that the status of the
            corresponding link qubits is updated appropriately.
            """

            def __init__(self, hyperedge: Hyperedge, servers: list[int]):
                self.hyperedge: Hyperedge = hyperedge
                # A dictionary of serverId to list of available link qubits.
                self.available: dict[int, list[Qubit]] = {s: [] for s in servers}
                # A dictionary of serverId to list of occupied link qubits.
                self.occupied: dict[int, list[Qubit]] = {s: [] for s in servers}
                # A dictionary of serverId to the link qubit holding a copy
                # of the qubit of ``hyperedge``.
                self.link_qubit_dict: dict[int, Qubit] = dict()
                # A dictionary of link qubits to link qubits. Whenever the
                # method `update_occupation` is called, the ID of the
                # input link qubit may change in order to avoid collision of
                # IDs. This dictionary keeps track of these changes,
                # mapping the old link qubit to the new one.
                #
                # NOTE: The entries of this dictionary may change within
                # the same call to `to_pytket_circuit_one_hyperedge`. This
                # is due to the fact that link qubits are reused and, at
                # different times in the circuit, different IDs may need
                # to be assigned in order to avoid collisions.
                self.link_qubit_id_update: dict[Qubit, Qubit] = dict()

            def _request_link_qubit(self, server: int) -> Qubit:
                """Returns an available link qubit in ``server``. If there are
                none, it creates a new one.
                Do not call this function outside the LinkManager class.

                :raise ConstraintException: If a server's communication
                capacity is exceeded and `satisfy_bound` was set to True.
                """
                if satisfy_bound and server_ebit_mem[server] > 0:
                    if server_ebit_mem[server] <= len(self.occupied[server]):
                        raise ConstraintException(
                            "Communication memory capacity of server "
                            f"{server} exceeded. \n\tConsider setting "
                            "allow_update to True: \n\t"
                            "to_pytket_circuit(allow_update=True)",
                            server,
                        )

                if not self.available[server]:
                    next_id = len(self.occupied[server])
                    qubit = Qubit(f"server_{server}_link_register", next_id)
                else:
                    qubit = self.available[server].pop()

                self.occupied[server].append(qubit)

                return qubit

            def _release_link_qubit(self, link_qubit: Qubit):
                """Releases ``link_qubit``, making it available again.
                Do not call this function outside the LinkManager class.
                """

                # Retrieve the server ``link_qubit`` is in
                server = get_server_id(link_qubit)

                # Make the link_qubit available
                self.occupied[server].remove(link_qubit)
                self.available[server].append(link_qubit)

            def start_link(self, target: int) -> list[EjppAction]:
                """Find sequence of StartingProcesses required to share the
                qubit of ``self.hyperedge`` with ``target`` server.
                """
                # NOTE: If all of the ``hyp_servers`` are to be connected,
                # then the routine below will "grow" the tree branch by
                # branch. And, since EJPP starting processes commute with each
                # other, the ordering of how it is "grown" does not matter.

                # Extract hyperedge data
                hyperedge = self.hyperedge
                q_vertex = hyp_circ.get_qubit_vertex(hyperedge)
                home_server = placement_map[q_vertex]
                hyp_servers = [placement_map[v] for v in hyperedge.vertices]
                tree = steiner_tree(server_graph, hyp_servers)
                assert target in hyp_servers

                # For each server connected to the qubit in ``hyperedge``,
                # find the shortest path to the target server and use the
                # one that is shortest among them
                connected_servers = self.connected_servers()
                connected_servers.append(home_server)
                best_path = None
                for c_server in connected_servers:
                    connection_path = nx.shortest_path(tree, c_server, target)
                    # fmt: off
                    if (
                        best_path is None or
                        len(connection_path) < len(best_path)
                    ):
                        best_path = connection_path
                    # fmt: on
                assert best_path is not None
                # NOTE: the ``best_path`` will only contain
                # one server from ``connected_servers``. Proof: if
                # it had two, the shortest path from the second
                # would be shorter => contradiction

                # Go over ``best_path`` step by step adding the required
                # starting EjppActions
                starting_actions = []
                source = best_path.pop(0)
                last_link_qubit = self.get_link_qubit(source)
                for next_server in best_path:
                    # Retrieve an available link qubit to populate
                    this_link_qubit = self._request_link_qubit(next_server)
                    self.link_qubit_dict[next_server] = this_link_qubit
                    # Add the EjppAction to create the connection
                    starting_actions.append(
                        EjppAction(
                            from_qubit=last_link_qubit,
                            to_qubit=this_link_qubit,
                        )
                    )
                    last_link_qubit = this_link_qubit

                return starting_actions

            def end_links(self, targets: list[int]) -> list[EjppAction]:
                """Find the sequence of EndingProcesses required to end the
                connection of the qubit in ``hyperedge`` to each server in
                ``targets``.
                """

                # Find the HW qubit holding the qubit in ``Hyperedge``
                hyperedge = self.hyperedge
                q_vertex = hyp_circ.get_qubit_vertex(hyperedge)
                circ_qubit = hyp_circ.get_qubit_of_vertex(q_vertex)
                home_link = qubit_mapping[circ_qubit]

                # Disconnect each server in ``targets``
                ending_actions = []
                for target in targets:
                    # Find the corresponding HW qubit
                    target_link = self.get_link_qubit(target)
                    # Disconnect ``target_link``
                    ending_actions.append(
                        EjppAction(from_qubit=target_link, to_qubit=home_link)
                    )
                    # Release the HW qubit acting as ``target_link``
                    self._release_link_qubit(target_link)
                    del self.link_qubit_dict[target]

                return ending_actions

            def connected_servers(self) -> list[int]:
                return list(self.link_qubit_dict.keys())

            def get_link_qubit(self, server: int) -> Qubit:
                """If ``server`` is the home server of the hyperedge's qubit,
                its hardware qubit is returned. Otherwise, we query the dict
                ``link_qubit_dict`` to retrieve the appropriate link qubit.
                """
                q_vertex = hyp_circ.get_qubit_vertex(self.hyperedge)
                circ_qubit = hyp_circ.get_qubit_of_vertex(q_vertex)
                if placement_map[q_vertex] == server:
                    return qubit_mapping[circ_qubit]
                else:
                    return self.link_qubit_dict[server]

            def update_occupation(self, link_qubit: Qubit, starting: bool):
                """Update the status of `self.occupied` and `self.available`
                for ``link_qubit``. This is meant to be used whenever an
                EJPP on ``link_qubit`` is encountered when scanning the
                circuit within `to_pytket_circuit_one_hyperedge`. The flag
                `starting` indicates if it is a starting or ending process.
                A link qubit is requested/released as usual, in order to
                make sure that the LinkManager is aware of all link qubits
                in the circuit and, hence, avoids collisions of IDs.
                This may mean that the ID of the link qubit is changed;
                this information is added to `link_qubit_id_update`.
                """
                if starting:
                    server = get_server_id(link_qubit)
                    new_link_qubit = self._request_link_qubit(server)
                    self.link_qubit_id_update[link_qubit] = new_link_qubit
                else:
                    new_link_qubit = self.link_qubit_id_update[link_qubit]
                    self._release_link_qubit(new_link_qubit)

            def get_updated_name(self, qubit: Qubit) -> Qubit:
                """Interface for `link_qubit_id_update`."""
                if qubit in self.link_qubit_id_update.keys():
                    return self.link_qubit_id_update[qubit]
                else:
                    return qubit

        def requires_embedding_start_proc(commands: list[Command]) -> bool:
            """Given a list of ``commands`` starting at a start_proc, check
            whether the start_proc's target link qubit has a CU1 gate acting
            on it before its end_proc.
            Otherwise, this start_proc is simply assisting entanglement
            swapping and does not need to be considered while embedding.
            """
            start_proc = commands[0]
            assert is_start_proc(start_proc)

            target_link = start_proc.qubits[1]
            requires_embedding = None
            for g in commands:
                if g.op.type == OpType.CU1 and target_link in g.qubits:
                    requires_embedding = True
                    break
                # Otherwise, stop when finding its end_proc
                elif is_end_proc(g) and target_link == g.qubits[0]:
                    requires_embedding = False
                    break

            assert requires_embedding is not None
            return requires_embedding

        def requires_embedding_end_proc(commands: list[Command]) -> bool:
            """Given a list of ``commands`` ending at an end_proc, check
            whether the end_proc's source link qubit has a CU1 gate acting
            on it after its start_proc.
            Otherwise, this end_proc is simply assisting entanglement
            swapping and does not need to be considered while embedding.
            """
            end_proc = commands[-1]
            assert is_end_proc(end_proc)

            source_link = end_proc.qubits[0]
            requires_embedding = None
            for g in reversed(commands):
                if g.op.type == OpType.CU1 and source_link in g.qubits:
                    requires_embedding = True
                    break
                # Otherwise, stop when finding its start_proc
                elif is_start_proc(g) and source_link == g.qubits[1]:
                    requires_embedding = False
                    break

            assert requires_embedding is not None
            return requires_embedding

        def to_pytket_circuit_one_hyperedge(
            hyperedge: Hyperedge, circ: Circuit
        ) -> Circuit:
            """Given a circuit equivalent to the original one, but with some
            of its non-local gates already distributed, implement those of the
            given hyperedge and return the new equivalent circuit.

            :raise ConstraintException: If a server's communication capacity
                is exceeded and `satisfy_bound` was set to True.
            """
            if len(hyperedge.vertices) == 1:  # Edge case: no gate vertices
                return circ

            new_circ = Circuit()
            for hw_qubit in qubit_mapping.values():
                new_circ.add_qubit(hw_qubit)

            # Extract hyperedge data
            q_vertex = hyp_circ.get_qubit_vertex(hyperedge)
            src_qubit = qubit_mapping[hyp_circ.get_qubit_of_vertex(q_vertex)]

            # Data to keep around during iterations
            #   `v_gate` stands for "gate vertex"; it keeps track of
            #   the vertex index of the last CU1 gate encountered while
            #   scanning the circuit
            v_gate = len(qubit_mapping) - 1
            #   `currently_h_embedding` is a boolean flag that indicates
            #   whether we are within an embedding unit at the moment
            currently_h_embedding = False
            #   `linkman` will help use manage the link qubits
            linkman = LinkManager(hyperedge, self.network.get_server_list())
            #   `carry_phase` is the phase pushed around within an embedding
            #   unit
            carry_phase = 0.0

            # Iterate over the commands of `circ`
            commands = circ.get_commands()
            for cmd_idx, cmd in enumerate(commands):
                # Keep track of how many of the original CU1 we've seen so far
                if cmd.op.type == OpType.CU1:
                    v_gate += 1
                # Keep track of the occupation of link qubits
                if is_start_proc(cmd):
                    try:
                        linkman.update_occupation(cmd.qubits[1], starting=True)
                    except ConstraintException as e:
                        e.v_gate = v_gate
                        raise e
                    remote_qubit = linkman.get_updated_name(cmd.qubits[1])
                    if remote_qubit not in new_circ.qubits:
                        new_circ.add_qubit(remote_qubit)
                if is_end_proc(cmd):
                    try:
                        linkman.update_occupation(cmd.qubits[0], starting=False)
                    except ConstraintException as e:
                        e.v_gate = v_gate
                        raise e

                # Trivial for every command that is not in the hyperedge's
                # subcircuit
                gate_vertices = hyp_circ.get_gate_vertices(hyperedge)
                # fmt: off
                if (
                    v_gate < min(gate_vertices) or  # Before hyperedge
                    v_gate >= max(gate_vertices) and cmd.op.type != OpType.CU1
                ):
                    qs = [linkman.get_updated_name(q) for q in cmd.qubits]
                    if cmd.op.type == OpType.Barrier:
                        new_circ.add_barrier(new_circ.qubits)
                    else:
                        new_circ.add_gate(cmd.op, qs)
                    continue
                # fmt: on

                # ~ Rz gate ~#
                if cmd.op.type == OpType.Rz:
                    # Trivial if not acting on `src_qubit`
                    if src_qubit not in cmd.qubits:
                        qs = [linkman.get_updated_name(q) for q in cmd.qubits]
                        new_circ.add_gate(cmd.op, qs)
                        continue

                    q = cmd.qubits[0]
                    phase = cmd.op.params[0]
                    # Append the gate
                    new_circ.Rz(phase, q)

                    # If not H-embedding, nothing needs to be done; otherwise:
                    if currently_h_embedding:
                        # The phase must be multiple of pi (either I or Z gate)
                        if isclose(phase % 1, 0) or isclose(phase % 1, 1):
                            # Apply it to all connected servers
                            for server in linkman.connected_servers():
                                link_qubit = linkman.link_qubit_dict[server]
                                new_circ.Rz(phase, link_qubit)
                        else:
                            # If it is not, we can try to fix it by carrying
                            # it after the next CU1 gate and try to cancel it.
                            #
                            # NOTE: We are assuming that this hyperedge can
                            #   be implemented using hopping packets; which
                            #   is something we check via PacMan. Hence, if
                            #   PacMan told us this is packing is valid, we
                            #   are guaranteed to be able to cancel this phase
                            carry_phase += phase

                # ~ H gate ~#
                elif cmd.op.type == OpType.H:
                    # Trivial if not acting on `src_qubit`
                    if src_qubit not in cmd.qubits:
                        qs = [linkman.get_updated_name(q) for q in cmd.qubits]
                        new_circ.add_gate(cmd.op, qs)
                        continue

                    q = cmd.qubits[0]
                    # The presence of an H gate indicates the beginning or end
                    # of an H-embedding on the qubit

                    if not currently_h_embedding:  # Starts an embedding unit
                        # There are two cases to consider:
                        #
                        # (Case 1) There is no CU1 gate or EJPP process that
                        # needs to be embedded.
                        # (Case 2) There is at least one.

                        found_embedded_cmd = None
                        i = cmd_idx + 1
                        for g in commands[(cmd_idx + 1) :]:  # noqa: E203
                            if g.op.type == OpType.CU1 and q in g.qubits:
                                found_embedded_cmd = g
                                break
                            elif (
                                is_start_proc(g)
                                and q == origin_of_start_proc(g, new_circ.qubits)
                                and requires_embedding_start_proc(commands[i:])
                            ):
                                found_embedded_cmd = g
                                break
                            elif (
                                is_end_proc(g)
                                and q == g.qubits[1]
                                and requires_embedding_end_proc(commands[: (i + 1)])
                            ):
                                found_embedded_cmd = g
                                break
                            # Otherwise, stop when finding an H gate on q
                            elif g.op.type == OpType.H and q in g.qubits:
                                break
                            i += 1

                        if found_embedded_cmd is None:  # (Case 1)
                            # Trivial: we simply need to apply H gate on the
                            # open link qubits
                            for server in linkman.connected_servers():
                                link_qubit = linkman.link_qubit_dict[server]
                                new_circ.H(link_qubit)
                            # Switch the value of the flag
                            currently_h_embedding = True

                        else:  # (Case 2)
                            # All connections to servers must be closed, except
                            # that of the remote server the CU1 gate acts on.
                            #
                            # NOTE: Due to the conditions of embeddability,
                            # embedded CU1 gates all act on the same servers

                            if is_start_proc(found_embedded_cmd):
                                remote_qubit = found_embedded_cmd.qubits[1]
                            else:
                                remote_qubit = [
                                    rq for rq in found_embedded_cmd.qubits if rq != q
                                ][0]
                            # In the case of ``found_embedded_cmd`` being a
                            # CU1 gate that has already been distributed,
                            # ``remote_qubit`` may be a link qubit, rather
                            # than the original one. The one that we actually
                            # want to use is the original one, since that's
                            # the one that was considered when calling
                            # ``hyperedge_cost``.
                            if (
                                found_embedded_cmd.op.type == OpType.CU1
                                and is_link_qubit(remote_qubit)
                            ):
                                # Find the next ending process on this
                                # ``remote_qubit``; it's target qubit
                                # will be the original qubit.
                                original_remote = None
                                for g in commands[i:]:
                                    if is_end_proc(g) and g.qubits[0] == remote_qubit:
                                        original_remote = g.qubits[1]
                                        break
                                assert original_remote is not None
                                assert not is_link_qubit(original_remote)
                                remote_qubit = original_remote

                            remote_server = get_server_id(remote_qubit)
                            # All servers but ``remote_server`` must be
                            # disconnected.
                            # Notice that it is not guaranteed nor necessary
                            # that ``remote_server`` is in the list of
                            # connected servers.
                            end_servers = [
                                s
                                for s in linkman.connected_servers()
                                if s != remote_server
                            ]

                            # Close the connections
                            for ejpp_end in linkman.end_links(end_servers):
                                new_circ.add_custom_gate(
                                    end_proc(),
                                    [],
                                    [ejpp_end.from_qubit, ejpp_end.to_qubit],
                                )

                            # Apply an H gate on the surviving link qubits
                            for server in linkman.connected_servers():
                                link_qubit = linkman.link_qubit_dict[server]
                                new_circ.H(link_qubit)
                            # Switch the value of the flag
                            currently_h_embedding = True

                    else:  # Currently embedding
                        # There are two cases to consider:
                        #
                        # (Case A) `carry_phase` is multiple of pi,
                        # (Case B) it is not.
                        #
                        # NOTE: In here we only worry about the link qubits.
                        # the 1-qubit operations on the original qubit are
                        # left unchanged.

                        # I gate, no need to apply Euler decomposition
                        if isclose(1 + carry_phase % 2, 1):  # (Case A)
                            # This means that H should end the embedding
                            currently_h_embedding = False
                            # Apply an H gate on all open link qubits
                            for server in linkman.connected_servers():
                                link_qubit = linkman.link_qubit_dict[server]
                                new_circ.H(link_qubit)

                        # Z gate, no need to apply Euler decomposition
                        elif isclose(carry_phase % 2, 1):  # (Case A)
                            # This means that H should end the embedding
                            currently_h_embedding = False
                            # Apply it to all connected servers
                            for server in linkman.connected_servers():
                                link_qubit = linkman.link_qubit_dict[server]
                                new_circ.Rz(carry_phase, link_qubit)
                                new_circ.H(link_qubit)
                            # Reset carry phase to zero
                            carry_phase = 0

                        # S or S' gate, apply Euler decomposition of H
                        elif isclose(carry_phase % 1, 0.5):  # (Case B)
                            # Replace HS with S'HS'H
                            # The middle S' is outside of an embedding unit
                            # since it is sandwiched by H gates. Hence, we
                            # must not copy it to the link qubit.
                            # Then, in the link qubit the H gates are
                            # contiguous, so they cancel each other.
                            # Consequently, we only need to carry the phase
                            # of the S' and push it later in the circuit
                            carry_phase = -carry_phase
                            # We are still embedding (we applied two H)
                            currently_h_embedding = True

                        # Other phases cannot be cancelled
                        else:
                            raise Exception(
                                "Implementation of hyperedge "
                                + f"{hyperedge.vertices} failed. It contains"
                                + "an H-type embedding with an internal phase"
                                + f"{carry_phase} that cannot be cancelled."
                                + "You should split this hyperedge into two."
                            )

                    # Append the original gate
                    new_circ.H(q)

                # ~ CU1 gate ~#
                elif cmd.op.type == OpType.CU1:
                    # Trivial if not acting on `src_qubit`
                    if src_qubit not in cmd.qubits:
                        qs = [linkman.get_updated_name(q) for q in cmd.qubits]
                        new_circ.add_gate(cmd.op, qs)
                        continue

                    phase = cmd.op.params[0]
                    rmt_candidates = [q for q in cmd.qubits if q != src_qubit]
                    assert len(rmt_candidates) == 1
                    rmt_qubit = rmt_candidates.pop()

                    # Check whether the gate is part of this hyperedge
                    if v_gate in hyperedge.vertices:  # Distribute it
                        # Sanity check: Not currently H embedding
                        assert not currently_h_embedding
                        # Find the server where the gate should be implemented
                        target_server = placement_map[v_gate]

                        # Add the required starting processes (if any)
                        try:
                            start_actions = linkman.start_link(target_server)
                        except ConstraintException as e:
                            e.v_gate = v_gate
                            raise e
                        for ejpp_start in start_actions:
                            if ejpp_start.to_qubit not in new_circ.qubits:
                                new_circ.add_qubit(ejpp_start.to_qubit)
                            new_circ.add_custom_gate(
                                start_proc(origin=src_qubit),
                                [],
                                [ejpp_start.from_qubit, ejpp_start.to_qubit],
                            )

                        # Append the gate to the circuit
                        new_circ.add_gate(
                            OpType.CU1,
                            phase,
                            [
                                linkman.get_link_qubit(target_server),
                                linkman.get_updated_name(rmt_qubit),
                            ],
                        )

                        # If this gate is the last gate in the hyperedge,
                        # end all connections.
                        if v_gate == max(hyperedge.vertices):
                            for ejpp_end in linkman.end_links(
                                linkman.connected_servers()
                            ):
                                new_circ.add_custom_gate(
                                    end_proc(),
                                    [],
                                    [ejpp_end.from_qubit, ejpp_end.to_qubit],
                                )

                    else:
                        qs = [linkman.get_updated_name(q) for q in cmd.qubits]
                        new_circ.add_gate(cmd.op, qs)

                        # Correction gates might need to be added
                        if currently_h_embedding:
                            # The phase must be multiple of pi
                            assert isclose(phase % 1, 0) or isclose(phase % 1, 1)

                            # If the CU1 gate has already been distributed,
                            # ``rmt_qubit`` may be a link qubit, rather
                            # than the original one. The one that we actually
                            # want to use is the original one, since that's
                            # the one that was considered when calling
                            # ``hyperedge_cost``.
                            if is_link_qubit(rmt_qubit):
                                # Find the next ending process on this
                                # ``remote_qubit``; it's target qubit
                                # will be the original qubit.
                                original_remote = None
                                for g in commands[cmd_idx:]:
                                    if is_end_proc(g) and g.qubits[0] == rmt_qubit:
                                        original_remote = g.qubits[1]
                                        break
                                assert original_remote is not None
                                rmt_qubit = original_remote
                            assert not is_link_qubit(rmt_qubit)

                            # A correction gate must be applied on every link
                            # qubit that is currently alive and has been used
                            # to implement this hyperedge.
                            for server in linkman.connected_servers():
                                link_qubit = linkman.get_link_qubit(server)
                                new_circ.add_gate(
                                    OpType.CZ,
                                    [
                                        link_qubit,
                                        rmt_qubit,
                                    ],
                                )
                            # CZ gates are used here to distinguish from CU1
                            # gates and, hence, do not mess with the counter
                            # `v_gate` in future calls to this function

                # ~ EJPP starting process ~#
                elif is_start_proc(cmd):
                    # Retrieve qubit information
                    orig_qubit = origin_of_start_proc(cmd, new_circ.qubits)
                    old_rmt_qubit = cmd.qubits[1]
                    remote_qubit = linkman.get_updated_name(old_rmt_qubit)
                    server = get_server_id(remote_qubit)

                    # Apply the command
                    qs = [linkman.get_updated_name(q) for q in cmd.qubits]
                    new_circ.add_gate(cmd.op, qs)

                    if currently_h_embedding and src_qubit == orig_qubit:
                        # A correction gate must be applied only if the
                        # start_proc must be considered when embedding.
                        if requires_embedding_start_proc(commands[cmd_idx:]):
                            # A correction gate must be applied on every link
                            # qubit that is currently alive and has been used
                            # to implement this hyperedge.
                            for server in linkman.connected_servers():
                                link_qubit = linkman.get_link_qubit(server)
                                new_circ.H(remote_qubit)
                                new_circ.CZ(remote_qubit, link_qubit)
                                new_circ.H(remote_qubit)

                # ~ EJPP ending process ~#
                elif is_end_proc(cmd):
                    # Retrieve qubit information
                    orig_qubit = linkman.get_updated_name(cmd.qubits[1])
                    remote_qubit = linkman.get_updated_name(cmd.qubits[0])
                    server = get_server_id(remote_qubit)

                    if currently_h_embedding and src_qubit == orig_qubit:
                        # A correction gate must be applied only if the
                        # start_proc must be considered when embedding.
                        if requires_embedding_end_proc(commands[: (cmd_idx + 1)]):
                            # A correction gate must be applied on every link
                            # qubit that is currently alive and has been used
                            # to implement this hyperedge.
                            for server in linkman.connected_servers():
                                link_qubit = linkman.get_link_qubit(server)
                                new_circ.H(remote_qubit)
                                new_circ.CZ(remote_qubit, link_qubit)
                                new_circ.H(remote_qubit)

                    # Apply the command
                    qs = [linkman.get_updated_name(q) for q in cmd.qubits]
                    new_circ.add_gate(cmd.op, qs)

                # ~ Extra stuff ~#
                # Extra CZ gates and Barriers are added to the circuit during
                # the circuit generation routine. These were not present in
                # the circuit the user inputted and they are removed or
                # replaced accordingly at the end of `to_pytket_circuit`.
                #
                # CZ gates correspond to correction gates due to H-embedding.
                # We use CZ instead of CU1 to make sure that `v_gate` is
                # only considering the CU1 gates originally in the circuit.
                # They will be replaced with CU1 gates on the final circuit.
                #
                # Barriers are added at the beginning of `to_pytket_circuit`
                # after each CU1 gate, with the intention that the ordering
                # of the CU1 gates remains unchanged throughout the routine.
                # If it changed, it would mess up the numbering of gate
                # vertices. Without barriers, every time we generate a new
                # equivalent circuit the ordering of parallel gates may
                # change (even though we insert them in the same order) due
                # to the internal workings of `pytket.Circuit`.
                elif cmd.op.type == OpType.CZ:
                    # Apply the command
                    qs = [linkman.get_updated_name(q) for q in cmd.qubits]
                    new_circ.add_gate(cmd.op, qs)
                elif cmd.op.type == OpType.Barrier:
                    # Apply the command
                    new_circ.add_barrier(new_circ.qubits)
                else:
                    raise Exception(f"Command {cmd.op} not supported.")

            return new_circ

        # ~ Main body ~#
        # Rename the circuit's qubits
        new_circ = Circuit()
        for hw_qubit in qubit_mapping.values():
            new_circ.add_qubit(hw_qubit)
        # Add barriers after each CU1 gate. This avoids issues with
        # pytket changing the order of CU1 gates as we modify the
        # circuit, which would be problematic for our gate vertex
        # numbering system.
        for cmd in hyp_circ._circuit.get_commands():
            qs = [qubit_mapping[q] for q in cmd.qubits]
            new_circ.add_gate(cmd.op, qs)
            if cmd.op.type == OpType.CU1:
                new_circ.add_barrier(new_circ.qubits)

        # Implement the hyperedges one by one
        try:
            for hedge in hyp_circ.hyperedge_list:
                new_circ = to_pytket_circuit_one_hyperedge(hedge, new_circ)
        except ConstraintException as e:
            # If the user refuses to change the distribution, raise exception.
            if not allow_update:
                raise e

            # Otherwise, we must split a hyperedge to attempt to reduce the
            # number of ebits alive at the point the constrained was violated.
            # To do so, find all hyperedges simultaneous to `e.v_gate`.
            # Filter out actual edges since these are just a qubit-vertex and
            # a gate-vertex and cannot be split.
            hedges = [
                h
                for h in hyp_circ.hyperedge_list
                if len(h.vertices) > 2
                and any(v <= e.v_gate for v in hyp_circ.get_gate_vertices(h))
                and any(v >= e.v_gate for v in hyp_circ.get_gate_vertices(h))
            ]

            # Choose the hyperedge from `hedges` that has the further distance
            # between its two vertices before and after `v_gate`.
            chosen_hedge = None
            longest_dist = -1
            for h in hedges:
                gates = hyp_circ.get_gate_vertices(h)
                # Get the first gate-vertex previous to `e.v_gate`
                v_prev = e.v_gate  # Default if none
                prev_vertices = [v for v in gates if v < e.v_gate]
                if prev_vertices:
                    v_prev = max(prev_vertices)
                # Get the first gate-vertex after `e.v_gate`
                v_post = e.v_gate  # Default if none
                post_vertices = [v for v in gates if v > e.v_gate]
                if post_vertices:
                    v_post = min(post_vertices)
                # Choose according to longest distance
                dist = v_post - v_prev
                if longest_dist < dist:
                    # Check that this hyperedge actually uses a link
                    # qubit on `e.server`.
                    h_servers = [placement_map[v] for v in h.vertices]
                    tree = steiner_tree(server_graph, h_servers)
                    # If it does, it's the best split so far, otherwise skip.
                    if e.server in tree.nodes:
                        longest_dist = dist
                        chosen_hedge = h

            # If no hyperedge could be chosen, inform the user
            if chosen_hedge is None:
                home_servers = [
                    placement_map[hyp_circ.get_qubit_vertex(h)]
                    for h in hyp_circ.get_hyperedges_containing([e.v_gate])
                ]
                placed_server = placement_map[e.v_gate]
                if placed_server not in home_servers:
                    # Then, `v_gate` is an evicted gate; it can only be
                    # implemented if chosen_server has more than 2 qubits
                    # of communication capacity.
                    if server_ebit_mem[placed_server] < 2:
                        raise ConstraintException(
                            f"Could not implement the gate {e.v_gate} since"
                            "it is placed on a non-home server "
                            f"{placed_server} and said server has memory "
                            "capacity less than two. Consider using an "
                            "approach that does not produce evicted gates, "
                            "or increase the communication memory of "
                            f"server {placed_server}.",
                            placed_server,
                        )
                # Otherwise, unexpected error
                raise ConstraintException(
                    "The distribution could not be amended to satisfy "
                    "the bound on communication capacity.",
                    e.server,
                )

            # Split the chosen hyperedge
            v_q = hyp_circ.get_qubit_vertex(chosen_hedge)
            gates = hyp_circ.get_gate_vertices(chosen_hedge)
            prev_vertices = [v for v in gates if v <= e.v_gate]
            post_vertices = [v for v in gates if v > e.v_gate]
            # If `post_vertices` is empty it can only be because the
            # last gate vertex in `chosen_hedges` is `e.v_gate`. Fix split.
            if not post_vertices:
                last_prev = prev_vertices.pop()
                assert last_prev == e.v_gate
                post_vertices = [last_prev]
            # Perform the split
            prev_hedge = Hyperedge([v_q] + prev_vertices)
            post_hedge = Hyperedge([v_q] + post_vertices)
            hyp_circ.split_hyperedge(
                old_hyperedge=chosen_hedge,
                new_hyperedge_list=[prev_hedge, post_hedge],
            )
            # Sanity check: the distribution is updated "in place"
            assert hyp_circ is self.circuit

            # Run to_pytket_circuit on the updated distribution
            return self.to_pytket_circuit(allow_update=allow_update)

        # Turn every CZ (correction) gate to CU1; remove barriers
        # Remove the origin qubit from the name of each start_proc
        final_circ = Circuit()
        for q in new_circ.qubits:
            final_circ.add_qubit(q)
        for cmd in new_circ.get_commands():
            if cmd.op.type == OpType.Barrier:
                continue
            elif cmd.op.type == OpType.CZ:
                final_circ.add_gate(OpType.CU1, 1.0, cmd.qubits)
            elif is_start_proc(cmd):
                final_circ.add_custom_gate(start_proc(), [], cmd.qubits)
            else:
                final_circ.add_gate(cmd.op, cmd.qubits)

        # Final sanity checks
        assert all_cu1_local(final_circ)
        assert check_equivalence(self.circuit.get_circuit(), final_circ, qubit_mapping)
        if ebit_cost(final_circ) != self.cost():
            detached_gates = self.detached_gate_list()
            embedded_gates = self.circuit.get_all_h_embedded_gate_vertices()
            # If a gate is both detached and embedded, the cost estimate may
            # be wrong. This should never happen using or current workflows
            # and it is here only for the sake of backwards compatibility.
            if any(detached in embedded_gates for detached in detached_gates):
                warnings.warn(
                    "Detected a gate that is both detached and embedded. "
                    + "As a consequence, the estimated cost did not match "
                    + f"actual cost. Estimate: {self.cost()}, "
                    + f"real: {ebit_cost(final_circ)}."
                )
            else:
                raise Exception(
                    "Estimated cost does not match actual cost. Estimate: "
                    + f"{self.cost()}, real: {ebit_cost(final_circ)}."
                )

        return final_circ
