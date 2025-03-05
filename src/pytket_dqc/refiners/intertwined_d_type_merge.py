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

from .refiner import Refiner
from pytket_dqc.circuits.distribution import Distribution
from pytket_dqc.allocators import GainManager
from pytket import OpType


class IntertwinedDTypeMerge(Refiner):
    """:class:`.Refiner` merging packets when they are intertwined. A packet is
    intertwined with another if the second packet contains gates which are
    intermittent between gates in the first.
    """

    def refine(self, distribution: Distribution, **kwargs) -> bool:
        """Merge intertwined packets.

        :param distribution: Distribution whose intertwined packets should
            be merged.
        :type distribution: Distribution
        :return: True if a refinement has been performed. False otherwise.
        :rtype: bool
        """

        gain_mgr = GainManager(initial_distribution=distribution)
        refinement_made = False

        # Iterate through all qubits, merging packets.
        for qubit_vertex in gain_mgr.distribution.circuit.get_qubit_vertices():
            # List of hyperedges acting on qubit
            hedge_list = gain_mgr.distribution.circuit.hyperedge_dict[
                qubit_vertex
            ].copy()

            # List of gate vertices belonging to all hyperedges in
            # hedge_list.
            gate_vertices_list = [
                gain_mgr.distribution.circuit.get_gate_vertices(hedge)
                for hedge in hedge_list
            ]

            while len(hedge_list) >= 2:
                first_hedge = hedge_list.pop(0)
                first_hedge_gate_vertices = gate_vertices_list.pop(0)

                assert len(hedge_list) == len(gate_vertices_list)

                # A list of all hyperedges which are intertwined with the
                # first. A hyperedge is intertwined if its first gate
                # appears in the circuit before the last of the original
                # packet
                intertwined = [
                    (hedge, gate_vertices)
                    for hedge, gate_vertices in zip(hedge_list, gate_vertices_list)
                    if min(gate_vertices) <= max(first_hedge_gate_vertices)
                ]

                # Iterate through intertwined hyperedge, until we find one
                # that can be merged. Do so if it is possible. All packets
                # which are merged are removed from hedge_list.
                for intertwined_hedge, gate_vertices in intertwined:
                    # find the pairs of gates, one in first_hedge and one
                    # in intertwined_hedge, with only gates not in
                    # either hyperedge between them. This is done by checking
                    # all gates in intertwined_hedge which are intermittent
                    # between consecutive gates in first_hedge, and taking
                    # the maximal and minimal indexes of such gates.
                    neighbour_gates = []
                    for i, j in zip(
                        first_hedge_gate_vertices[:-1], first_hedge_gate_vertices[1:]
                    ):
                        # List of gates in gate_vertices that are between
                        # consecutive gates in first_hedge_gate_vertices.
                        intermittent_gates = [k for k in gate_vertices if i < k < j]

                        # If there are gates intermittent between gates in
                        # first_hedge_gate_vertices store the indices of those
                        # pairs of neighbouring gates.
                        if len(intermittent_gates) > 0:
                            neighbour_gates.extend(
                                [
                                    (i, min(intermittent_gates)),
                                    (max(intermittent_gates), j),
                                ]
                            )

                    intermediate_commands_list = [
                        gain_mgr.distribution.circuit.get_intermediate_commands(  # noqa: E501
                            first_vertex=gate_one,
                            second_vertex=gate_two,
                            qubit_vertex=qubit_vertex,
                        )
                        for gate_one, gate_two in neighbour_gates
                    ]

                    # If any of the subcircuits between neighbouring gates
                    # do not contain H gates then the hyperedges can be
                    # merged. Remove the packets with which the original
                    # packet is being merged.
                    if any(
                        all(command.op.type != OpType.H for command in commands_list)
                        for commands_list in intermediate_commands_list
                    ):
                        first_hedge = gain_mgr.merge_hyperedge(
                            [first_hedge, intertwined_hedge]
                        )
                        del gate_vertices_list[hedge_list.index(intertwined_hedge)]
                        del hedge_list[hedge_list.index(intertwined_hedge)]
                        refinement_made = True

        assert gain_mgr.distribution.is_valid()
        return refinement_made
