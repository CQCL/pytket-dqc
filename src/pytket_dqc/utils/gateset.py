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

import numpy as np
import logging

from pytket.predicates import (
    GateSetPredicate,
    NoSymbolsPredicate,
    UserDefinedPredicate,
)

from pytket import OpType, Circuit, Qubit
from pytket.passes import (
    CustomPass,
    EulerAngleReduction,
    RebaseCustom,
    SquashCustom,
    RemoveRedundancies,
    SequencePass,
    BasePass,
)
from pytket.circuit import CustomGateDef, Op, Command
from typing import Optional
import sympy  # type: ignore

logging.basicConfig(level=logging.INFO)

#: Allowed gateset for distributors in pytket-dqc
dqc_1_qubit = {
    OpType.Rz,
    OpType.H,
}
dqc_2_qubit = {
    OpType.CU1,
}
dqc_gateset = dqc_1_qubit.union(dqc_2_qubit)


def check_function(circ):
    return NoSymbolsPredicate().verify(circ) and GateSetPredicate(dqc_gateset).verify(
        circ
    )


#: Predicate for checking gateset is valid
dqc_gateset_predicate = UserDefinedPredicate(check_function)


def cz_to_cu1(circ: Circuit) -> Circuit:
    """Convert all CZ gates in the circuit to CU1 gates."""
    new_circ = Circuit()
    for q in circ.qubits:
        new_circ.add_qubit(q)
    for cmd in circ.get_commands():
        if cmd.op.type == OpType.CZ:
            new_circ.add_gate(OpType.CU1, 1.0, cmd.qubits)
        else:
            new_circ.add_gate(cmd.op, cmd.qubits)
    return new_circ


def tk2_to_cu1(a, b, c) -> Circuit:
    """Given a TK2 gate XXPhase(a)*YYPhase(b)*ZZPhase(c), return
    an equivalent circuit using CU1 and single qubit gates.

    Note: Unfortunately, pytket does not currently support a simple
    interface to write rebase passes other than those based on replacing
    TK2 gates and CX gates; in this case, we are using the former.
    """
    circ = Circuit(2)
    # The ZZPhase(c) gate
    circ.add_gate(OpType.CU1, -2 * c, [0, 1]).Rz(c, 0).Rz(c, 1)
    # The YYPhase(b) gate
    circ.Sdg(0).Sdg(1).H(0).H(1).add_gate(OpType.CU1, -2 * b, [0, 1]).Rz(b, 0).Rz(
        b, 1
    ).H(0).H(1).S(0).S(1)
    # The XXPhase(a) gate
    circ.H(0).H(1).add_gate(OpType.CU1, -2 * a, [0, 1]).Rz(a, 0).Rz(a, 1).H(0).H(1)
    # The global phase (we could ignore it, but TKET lets us track it)
    circ.add_phase((a + b + c) / 2)
    return circ


def tk1_to_euler(a, b, c) -> Circuit:
    """Given a TK1 gate Rz(a)*Rx(b)*Rz(c), return an equivalent circuit
    using Rz and Rx gates.
    """
    # NOTE: The correctness of these gate replacements has been checked by
    # composing the new circuit with the adjoint of the original one. Such a
    # test was done for each of the cases (with appropriate values of ``b``)
    # where ``a`` and ``c`` were chosen at random.
    #
    # NOTE: Every sequence of gates that could be written down as a Rz*H*Rz
    # will be written down as such. To prove this we need to check that such
    # sequences of gates would always be represented by an Euler decomposition
    # where the Rx gate has phase pi/2 or -pi/2. The argument is as follows:
    #  - if it can be written as Rz*H*Rz it means that the absolute value of
    #    each entry in the matrix is 1/sqrt(2);
    #  - in the Euler decomposition, only Rx changes the absolute value of
    #    entries; in particular, for absolute value 1/sqrt(2) it must be
    #    either pi/2 or -pi/2.
    # We then use the decomposition of H = Rz(0.5)*Rx(0.5)*Rz(0.5) and
    # H = Rz(-0.5)*Rx(-0.5)*Rz(-0.5) to introduce the H gates as needed.

    if any(type(x) is sympy.core.mul.Mul for x in [a, b, c]):
        raise Exception("Symbolic parameters are not supported")

    circ = Circuit(1)

    # Case 1: the Rx gate can be removed
    if np.isclose(b % 2, 0) or np.isclose(b % 2, 2):
        circ.Rz(c + a, 0)
        if np.isclose(b % 4, 2):
            circ.add_phase(1.0)
    # Case 2: the Rx gate has a multiple of pi/2 phase
    elif np.isclose(b % 2, 0.5) or np.isclose(b % 2, 1.5):
        circ.Rz(c - b, 0).H(0).Rz(a - b, 0).add_phase(-0.5)
        if b % 4 > 2:
            circ.add_phase(1.0)
    # Case 3: for any other case, use Euler decomposition
    else:
        circ.Rz(c, 0).H(0).Rz(b, 0).H(0).Rz(a, 0)

    return circ


def to_euler_with_two_hadamards(ops: list[Op]) -> list[Op]:
    """Take a list of single qubit gates and substitute
    it with an equivalent list (up to global phase)
    that is of the form [Rz, H, Rz, H, Rz].

    NOTE: Global Phases are not preserved.
    """

    hadamard_indices = [i for i, op in enumerate(ops) if op.type == OpType.H]
    id_rz = Op.create(OpType.Rz, [0])
    hadamard = Op.create(OpType.H)
    hadamard_count = len(hadamard_indices)

    # The following should be guranteed by DQCPass()
    assert hadamard_count <= 2, f"There should not be more than 2 Hadamards. {ops}"

    new_ops: list[Op] = []

    # Insert I as appropriate
    if hadamard_count == 2:
        if ops[0].type == OpType.H:
            new_ops.append(id_rz)

        new_ops.extend(ops)

        if ops[-1].type == OpType.H:
            new_ops.append(id_rz)

    # Can just stick two at the ends
    elif hadamard_count == 0:
        if len(ops) == 0:
            new_ops.append(id_rz)
        new_ops.extend(ops)
        new_ops.append(hadamard)
        new_ops.append(id_rz)
        new_ops.append(hadamard)
        new_ops.append(id_rz)

    else:
        assert len(ops) <= 3, "There can only be up to 3 ops in this decomposition."
        s_op = Op.create(OpType.Rz, [0.5])

        # The list is just [H]
        if len(ops) == 1:
            new_ops += [s_op, hadamard, s_op, hadamard, s_op]

        # The list is [H, Op] or [Op, H]
        elif len(ops) == 2:
            phase_op_index = int(not hadamard_indices[0])  # only takes value of 1 or 0
            phase_op = ops[phase_op_index]
            phase = phase_op.params[0]  # phase in turns of pi
            new_phase_op = Op.create(
                OpType.Rz, [phase + 1 / 2]
            )  # need to add another half phase
            new_ops += [new_phase_op, hadamard, s_op, hadamard, s_op]
            if phase_op_index:
                new_ops.reverse()

        # List is [Op, H, Op]
        else:
            first_phase = ops[0].params[0]
            second_phase = ops[2].params[0]
            first_new_phase_op = Op.create(OpType.Rz, [first_phase + 0.5])
            second_new_phase_op = Op.create(OpType.Rz, [second_phase + 0.5])
            new_ops += [
                first_new_phase_op,
                hadamard,
                s_op,
                hadamard,
                second_new_phase_op,
            ]

    logging.debug(f"Converted {ops} for {new_ops}")

    assert len(new_ops) == 5
    assert all(
        [
            new_ops[0].type == OpType.Rz,
            new_ops[1].type == OpType.H,
            new_ops[2].type == OpType.Rz,
            new_ops[3].type == OpType.H,
            new_ops[4].type == OpType.Rz,
        ]
    )
    return new_ops


#: Pass rebasing gates to those valid within pytket-dqc
def DQCPass() -> BasePass:
    """Transpile a given circuit to the gateset supported by pytket-dqc
    by calling ``DQCPass().apply(circuit)``.
    """
    return SequencePass(
        [
            CustomPass(cz_to_cu1),
            RebaseCustom(dqc_gateset, tk2_to_cu1, tk1_to_euler),
            SquashCustom(dqc_1_qubit, tk1_to_euler),
            EulerAngleReduction(p=OpType.Rz, q=OpType.Rx),
            RemoveRedundancies(),
        ]
    )


#: Defining starting_process and ending_process custom gates
def_circ = Circuit(2)
def_circ.add_barrier([0, 1])


def start_proc(origin: Optional[Qubit] = None) -> CustomGateDef:
    if origin is None:
        name = "starting_process"
    else:
        name = "starting_process_" + str(origin)
    return CustomGateDef.define(name, def_circ, [])


def end_proc() -> CustomGateDef:
    return CustomGateDef.define("ending_process", def_circ, [])


def telep_proc() -> CustomGateDef:
    return CustomGateDef.define("teleportation", def_circ, [])


def is_start_proc(cmd: Command) -> bool:
    return cmd.op.type == OpType.CustomGate and cmd.op.get_name().startswith(
        "starting_process"
    )


def is_end_proc(cmd: Command) -> bool:
    return cmd.op.type == OpType.CustomGate and cmd.op.get_name().startswith(
        "ending_process"
    )


def is_telep_proc(cmd: Command) -> bool:
    return cmd.op.type == OpType.CustomGate and cmd.op.get_name().startswith(
        "teleportation"
    )


def origin_of_start_proc(cmd: Command, all_qubits: list[Qubit]) -> Qubit:
    """Not the `cmd.qubits[0]` of the start_proc, but the qubit that
    originally held the information being "copied" by this start_proc.
    NOTE: Since the qubit ID is stored as a string within the name of the
    CustomGate describing the start_proc, we need ``all_qubits`` from the
    circuit to find out which among them has the same ID and return that it.
    """

    assert is_start_proc(cmd)
    name = cmd.op.get_name()

    # Assume this is called only when the origin qubit has been
    # recorded when constructing this start_proc.
    assert name.startswith("starting_process_")

    qubit_str = name[len("starting_process_") :]
    potential_qubits = [q for q in all_qubits if str(q) == qubit_str]

    assert len(potential_qubits) == 1
    return potential_qubits[0]
