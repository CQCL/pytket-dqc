from pytket import Circuit, OpType
from pytket.passes import auto_rebase_pass
from .circuit_analysis import is_link_qubit

from pytket.extensions.pyzx import tk_to_pyzx  # type: ignore
import pyzx.circuit as zx  # type: ignore


def check_equivalence(circ1: Circuit, circ2: Circuit) -> bool:
    """
    Use PyZX to check the two circuits are equivalent. This is done by
    concatenating the ZX diagram of ``circ1`` with the adjoint of the
    ZX diagram of ``circ2`` to see if the result is the identity.

    Note: Since pytket-pyzx cannot deal with initial states and projections,
    we instead leave the ancilla qubits used in EJPP processes both
    unprepared and unmeasured and add extra wires on the other circuit as
    necessary so that both have the same width.
    """
    width = max(len(circ1.qubits), len(circ2.qubits))
    zx1 = to_pyzx(circ1, width)
    zx2 = to_pyzx(circ2, width)
    return zx1.verify_equality(zx2)


def to_pyzx(circuit: Circuit, n_wires: int) -> zx.Circuit:
    """Convert a circuit to a ZX diagram in PyZX. Every starting EJPP
    process and ending EJPP process is converted to a CX gate with control
    on the shared qubit and target on the ancilla qubit; no initial state nor
    projection is included: each ancilla qubit's wire exist from beginning to
    end.
    If ``n_wires`` is larger than the total number of qubits in the circuit,
    the circuit is tensored by the corresponding number of indentity wires so
    that the output circuit has the appropriate width.

    Note: Using CX gates to represent the start/end of an EJPP process is a
    valid simplification of the actual circuits that would be required (which
    involve Bell pairs and classically-controlled corrections. To be precise,
    state preparation and discarding would need to be included. Since
    discarding is not supported by PyZX (only projection), we instead keep
    ancilla qubits unprepared/unmeasured; this is a reliable way to check for
    equality without checking each possible measurement projection separately.
    """
    assert len(circuit.qubits) <= n_wires

    # To convert the circuit to a "simple" one (required by pytket-pyzx) we
    # need to figure out how to swap "link" qubits to the bottom so that we
    # need not distinguishing between "link" qubits and "workspace" qubits
    workspace_qubits = []
    link_qubits = []
    for q in circuit.qubits:
        if is_link_qubit(q):
            link_qubits.append(q)
        else:
            workspace_qubits.append(q)
    qubit_dict = {q: n for n, q in enumerate(workspace_qubits + link_qubits)}

    # Create the circuit by adding the swaps and rebasing every CustomGate
    new_circ = Circuit(n_wires)
    for command in circuit.get_commands():
        qubits = [qubit_dict[q] for q in command.qubits]

        if command.op.type == OpType.CustomGate:
            if command.op.get_name() == "StartingProcess":
                new_circ.CX(qubits[0], qubits[1])
            elif command.op.get_name() == "EndingProcess":
                new_circ.CX(qubits[1], qubits[0])
            else:
                raise Exception(
                    f"CustomGate {command.op.get_name()} not supported!"
                )

        else:
            new_circ.add_gate(command.op, qubits)

    # Rebase the circuit to a gateset that pytket-pyzx can handle
    pyzx_gateset = {OpType.Rz, OpType.Rx, OpType.CX}
    auto_rebase_pass(pyzx_gateset).apply(new_circ)

    return tk_to_pyzx(new_circ)
