import numpy as np  # type: ignore
from pytket.predicates import (  # type: ignore
    GateSetPredicate,
    NoSymbolsPredicate,
    UserDefinedPredicate,
)

from pytket import OpType, Circuit
from pytket.passes import (  # type: ignore
    EulerAngleReduction,
    RebaseCustom,
    SquashCustom,
    RemoveRedundancies,
    SequencePass,
)
from pytket.circuit import CustomGateDef  # type: ignore

#: Allowed gateset for distributors in pytket-dqc
dqc_1_qubit = {
    OpType.Rz,
    OpType.H,
    OpType.X,
    OpType.Z,
}
dqc_2_qubit = {
    OpType.CRz,
}
dqc_gateset = dqc_1_qubit.union(dqc_2_qubit)


def check_function(circ):

    return NoSymbolsPredicate().verify(circ) and GateSetPredicate(
        dqc_gateset
    ).verify(circ)


#: Predicate for checking gateset is valid
dqc_gateset_predicate = UserDefinedPredicate(check_function)


def tk2_to_crz(a, b, c) -> Circuit:
    """Given a TK2 gate XXPhase(a)*YYPhase(b)*ZZPhase(c), return
    an equivalent circuit using CRz and single qubit gates.

    Note: Unfortunately, pytket does not currently support a simple
    interface to write rebase passes other than those based on replacing
    TK2 gates and CX gates; in this case, we are using the former.
    """
    circ = Circuit(2)
    # The ZZPhase(c) gate
    circ.CRz(-2 * c, 0, 1).Rz(c, 1)
    # The YYPhase(b) gate
    circ.Sdg(0).Sdg(1).H(0).H(1).CRz(-2 * b, 0, 1).Rz(b, 1).H(0).H(1).S(0).S(1)
    # The XXPhase(a) gate
    circ.H(0).H(1).CRz(-2 * a, 0, 1).Rz(a, 1).H(0).H(1)
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
    #
    if np.isclose(b % 2, 0) or np.isclose(b % 2, 2):
        return Circuit(1).Rz(c + a, 0)
    if np.isclose(b % 2, 1):
        return Circuit(1).Rz(c, 0).X(0).Rz(a, 0)
    if np.isclose(b % 2, 0.5):
        return Circuit(1).Rz(c - 0.5, 0).H(0).Rz(a - 0.5, 0)
    if np.isclose(b % 2, 1.5):
        return Circuit(1).Rz(c + 0.5, 0).H(0).Rz(a + 0.5, 0)
    else:
        return Circuit(1).Rz(c, 0).H(0).Rz(b, 0).H(0).Rz(a, 0)


#: Pass rebasing gates to those valid within pytket-dqc
dqc_rebase = SequencePass(
    [
        RebaseCustom(dqc_gateset, tk2_to_crz, tk1_to_euler),
        SquashCustom(dqc_1_qubit, tk1_to_euler),
        EulerAngleReduction(p=OpType.Rz,q=OpType.Rx),
        RemoveRedundancies()
    ]
)

#: Defining StartingProcess and EndingProcess custom gates
def_circ = Circuit(2)
def_circ.add_barrier([0, 1])

start_proc = CustomGateDef.define("StartingProcess", def_circ, [])
end_proc = CustomGateDef.define("EndingProcess", def_circ, [])
telep_proc = CustomGateDef.define("Teleportation", def_circ, [])
