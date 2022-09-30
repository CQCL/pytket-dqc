from pytket import Circuit, OpType, Qubit
from pytket.passes import auto_rebase_pass

from pytket.extensions.pyzx import tk_to_pyzx  # type: ignore
import pyzx as zx  # type: ignore


def check_equivalence(
    circ1: Circuit, circ2: Circuit, qubit_mapping: dict[Qubit, Qubit]
) -> bool:
    """
    Use PyZX to check the two circuits are equivalent. This is done by
    concatenating the ZX diagram of ``circ1`` with the adjoint of the
    ZX diagram of ``circ2`` to see if the result is the identity.

    NOTE: returning False does not guarantee that the circuits are different;
    it just means that PyZX could not prove that they are equivalent.
    If it returns True and the ``qubit_mapping`` covers all qubits of both
    circuits, then it is guaranteed that the circuits are equivalent.
    If some qubits are missing from ``qubit_mapping`` then there are
    post-selections/projections to state 0, which means that the test is not
    a formal proof -- we would need to test for projection to state 1 as well
    (and, in fact, all combinations of 0 and 1).
    Hence, in the latter case, we can only say "we have strong evidence that
    the circuits are equivalent".

    :param circ1: The first of the two circuits to be compared for equality
    :type circ1: Circuit
    :param circ2: The second of the two circuits to be compared for equality
    :type circ2: Circuit
    :param qubit_mapping: A mapping from qubits of ``circ1`` to qubits of
        ``circ2``. If a qubit is not included in this dictionary it means that
        it ought to be treated as an ancilla (i.e. prepared and measured).
    :type qubit_mapping: dict[Qubit, Qubit]
    """

    # Note: the implementation is based on the code for verify_equality from
    # https://github.com/Quantomatic/pyzx/blob/master/pyzx/circuit/__init__.py
    # altered so that it works with zx.Graph instead of zx.Circuit.
    qubits1 = list(qubit_mapping.keys())
    qubits2 = list(qubit_mapping.values())
    # Check that the ``qubit_mapping`` is 1-to-1
    if len(list(set(qubits2))) != len(qubits2):
        raise Exception("The qubit mapping must be 1-to-1.")

    zx1 = to_pyzx(circ1, qubits1)
    zx2 = to_pyzx(circ2, qubits2)

    # Compose the adjoint of zx1 with zx2 and simplify
    g = zx1.adjoint()
    g.compose(zx2)
    zx.full_reduce(g)
    # Check that the only vertices that remain in the graph are those of
    # input and output per wire.
    # To make sure that g has no swaps, we check that the input and
    # output vertices are connected in the right order
    return g.num_vertices() == 2 * len(g.inputs()) and all(
        g.connected(v, w) for v, w in zip(g.inputs(), g.outputs())
    )


def to_pyzx(circuit: Circuit, mask: list[Qubit]) -> zx.Graph:
    """Convert a circuit to a ZX diagram in PyZX. Every starting EJPP
    process and ending EJPP process is converted to a CX gate. All non-ancilla
    qubits should be in ``mask`` and no ancilla qubits should be in ``mask``.

    Note: Ancilla qubits are projected to state 0. This is not equivalent to
    and EJPP ending process, since what we really need is to discard the qubit
    so we should check for every combination of projections to 0 and 1 states.
    However, projecting to 0 is enough for our purposes to give strong evidence
    of circuit equality.

    :param circuit: The circuit to be converted to a ZX-diagram
    :type circuit: Circuit
    :param mask: The list of qubits that are not ancillas, ordered according
        to the intended order of wires in the output ZX-diagram.
    :type mask: list[Qubit]
    """

    # We need that the logical qubits in ``mask`` are on the top
    # wires of the circuit, ordered as in ``mask``.
    # To do so, we figure out a dictionary of qubits to positions.
    qubit_dict = {q: n for n, q in enumerate(mask)}
    qubit_counter = len(qubit_dict)

    # We now check that the qubits not in ``mask`` are all ancilla qubits
    # "created" by a StartingProcess
    n_ebits = 0
    ancillas = set()
    for command in circuit.get_commands():
        if command.op.type == OpType.CustomGate:
            if command.op.get_name() == "StartingProcess":
                ancillas.add(command.qubits[1])
                n_ebits += 1
    if set(circuit.qubits) - set(mask) != ancillas:
        raise Exception("Violated: q in mask <=> q not ancilla")

    # Create the body of the circuit by rebasing every CustomGate
    the_circ = Circuit(len(mask) + n_ebits)
    for command in circuit.get_commands():
        if command.op.type == OpType.CustomGate:

            if command.op.get_name() == "StartingProcess":
                # Identify the qubit to share and create a new ancilla qubit
                if command.qubits[0] not in qubit_dict.keys():
                    raise Exception("Attempting to act on a discarded qubit")
                qubit_to_share = qubit_dict[command.qubits[0]]
                new_ancilla_qubit = qubit_counter
                # Update the qubit_counter, making room for the next ebit
                qubit_counter += 1
                # Apply the CX on ancilla (equivalent to a starting process)
                the_circ.CX(qubit_to_share, new_ancilla_qubit)
                # Update the qubit_dict to include the newly created ancilla
                qubit_dict[command.qubits[1]] = new_ancilla_qubit

            elif command.op.get_name() == "EndingProcess":
                # Identify the qubits
                if not all(q in qubit_dict.keys() for q in command.qubits):
                    raise Exception("Attempting to act on a discarded qubit")
                ancilla_qubit = qubit_dict[command.qubits[0]]
                qubit_shared = qubit_dict[command.qubits[1]]
                # Apply a CX; for this to be equivalent to an ending process
                # we would need to discard the ancilla qubit after the CX.
                # PyZX doesn't support simplification of diagrams with discard
                # so, instead, we will project this ancilla at the end.
                # In principle, we should project both to 0 and to 1; but
                # we will only project to 0; see the note in the docstring.
                the_circ.CX(qubit_shared, ancilla_qubit)
                # We no longer wish to apply more operations on the ancilla.
                # For the sake of sanity checks, we remove it from the
                # qubit_dict so that, if we attempt to apply an operation on
                # it, an exception will be raised.
                del qubit_dict[command.qubits[0]]

            else:
                raise Exception(
                    f"CustomGate {command.op.get_name()} not supported!"
                )
        else:
            if not all(q in qubit_dict.keys() for q in command.qubits):
                raise Exception("Attempting to act on a discarded qubit")
            qubits = [qubit_dict[q] for q in command.qubits]
            the_circ.add_gate(command.op, qubits)

    # At the end of the loop it should be satisfied that:
    assert qubit_counter == len(mask) + n_ebits

    # Rebase body_circ to a gateset that pytket-pyzx can handle
    pyzx_gateset = {OpType.H, OpType.Rz, OpType.Rx, OpType.CZ, OpType.CX}
    auto_rebase_pass(pyzx_gateset).apply(the_circ)
    zx_graph = tk_to_pyzx(the_circ).to_graph()

    # Add states 0 on link qubits and projections to state 0. To do so we
    # create a string that indicates what to do per qubit; '/' is "do-nothing"
    # '0' is "place a 0 state/effect"
    values = "/" * len(mask) + "0" * n_ebits
    zx_graph.apply_state(values)
    zx_graph.apply_effect(values)

    assert len(zx_graph.inputs()) == len(mask)
    assert len(zx_graph.outputs()) == len(mask)
    return zx_graph
