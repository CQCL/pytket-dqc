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
    assert len(list(set(qubits2))) == len(qubits2)

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
    process and ending EJPP process is converted to a CX gate. The ancilla
    qubits should not be inside ``mask``, so that they are initialised to
    state 0 and projected to 0 at the end of the circuit.

    Note: This is not equivalent, since what we really need is a discard not
    a projection (and if we use projections, we should check each of them).
    However, this is enough for our purposes to give strong evidence of
    circuit equality.

    Note: Instead of initialising and projecting the auxiliary qubit on each
    starting/ending process, we keep the wire alive for the whole duration of
    the circuit and only initialise and project at the two ends.

    :param circuit: The circuit to be converted to a ZX-diagram
    :type circuit: Circuit
    :param mask: The list of qubits that are not ancillas, ordered according
        to the intended order of wires in the output ZX-diagram.
    :type mask: list[Qubit]
    """

    # We need that the logical qubits in ``mask`` are on the top
    # wires of the circuit, ordered as in ``mask``.
    # To do so, we figure out a dictionary of qubits to positions.
    omitted = []
    for q in circuit.qubits:
        if q not in mask:
            omitted.append(q)
    qubit_dict = {q: n for n, q in enumerate(mask + omitted)}

    # Create the body of the circuit by rebasing every CustomGate
    the_circ = Circuit(circuit.n_qubits)
    for command in circuit.get_commands():
        qubits = [qubit_dict[q] for q in command.qubits]

        if command.op.type == OpType.CustomGate:
            if command.op.get_name() == "StartingProcess":
                the_circ.CX(qubits[0], qubits[1])
            elif command.op.get_name() == "EndingProcess":
                the_circ.CX(qubits[1], qubits[0])
            else:
                raise Exception(
                    f"CustomGate {command.op.get_name()} not supported!"
                )
        else:
            the_circ.add_gate(command.op, qubits)

    # Rebase body_circ to a gateset that pytket-pyzx can handle
    pyzx_gateset = {OpType.H, OpType.Rz, OpType.Rx, OpType.CZ, OpType.CX}
    auto_rebase_pass(pyzx_gateset).apply(the_circ)
    zx_graph = tk_to_pyzx(the_circ).to_graph()

    # Add states 0 on link qubits and projections to state 0. To do so we
    # create a string that indicates what to do per qubit; '/' is "do-nothing"
    # '0' is "place a 0 state/effect"
    values = "/" * len(mask) + "0" * len(omitted)
    zx_graph.apply_state(values)
    zx_graph.apply_effect(values)

    assert len(zx_graph.inputs()) == len(mask)
    assert len(zx_graph.outputs()) == len(mask)
    return zx_graph
