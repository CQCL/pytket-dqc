from pytket import Circuit, OpType
from pytket.passes import auto_rebase_pass
from .circuit_analysis import is_link_qubit

from pytket.extensions.pyzx import tk_to_pyzx  # type: ignore
import pyzx as zx  # type: ignore


def check_equivalence(circ1: Circuit, circ2: Circuit) -> bool:
    """
    Use PyZX to check the two circuits are equivalent. This is done by
    concatenating the ZX diagram of ``circ1`` with the adjoint of the
    ZX diagram of ``circ2`` to see if the result is the identity.

    Note: the implementation is based on the code for verify_equality from
    https://github.com/Quantomatic/pyzx/blob/master/pyzx/circuit/__init__.py
    altered so that it works with zx.Graph instead of zx.Circuit.
    """
    zx1 = to_pyzx(circ1)
    zx2 = to_pyzx(circ2)
    # Check that the number of workspace qubits match
    if len(zx1.inputs()) != len(zx2.inputs()):
        return False

    # Compose the adjoint of zx1 with zx2 and simplify
    g = zx1.adjoint()
    g.compose(zx2)
    zx.full_reduce(g)
    # Check that the only vertices that remain in the graph are those of
    # input and output per wire.
    # To make sure that the g has not introduced any swaps, we check that
    # the input and output vertices are connected in the right order
    return g.num_vertices() == 2 * len(g.inputs()) and all(
        g.connected(v, w) for v, w in zip(g.inputs(), g.outputs())
    )


def to_pyzx(circuit: Circuit) -> zx.Graph:
    """Convert a circuit to a ZX diagram in PyZX. Every starting EJPP
    process and ending EJPP process is converted to a CX gate with an initial
    state 0 for starting processes and projection to 0 for ending processes.

    Note: This is not equivalent, since what we really need is a discard not
    a projection (and if we use projections, we should check each of them).
    However, this is enough for our purposes to give strong evidence of
    circuit equality.

    Note: Instead of initialising and projecting the aunxiliary qubit on each
    starting/ending process, we keep the wire alive for the whole duration of
    the circuit and only initialise and project at the two ends.
    """

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

    # Create the body of the circuit by rebasing every CustomGate
    the_circ = Circuit(len(qubit_dict))
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
    values = "/" * len(workspace_qubits) + "0" * len(link_qubits)
    zx_graph.apply_state(values)
    zx_graph.apply_effect(values)

    assert len(zx_graph.inputs()) == len(workspace_qubits)
    assert len(zx_graph.outputs()) == len(workspace_qubits)
    return zx_graph
