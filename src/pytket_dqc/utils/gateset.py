from pytket.predicates import (  # type: ignore
    GateSetPredicate,
    NoSymbolsPredicate,
    UserDefinedPredicate,
)

from pytket import OpType, Circuit
from pytket.passes import (  # type: ignore
    RebaseCustom,
)
from pytket.circuit import CustomGateDef  # type: ignore

#: Allowed gateset for distributors in pytket-dqc
dqc_gateset = {
    OpType.Rz,
    OpType.X,
    OpType.H,
    OpType.CRz,
    OpType.Measure,
}


def check_function(circ):

    return NoSymbolsPredicate().verify(circ) and GateSetPredicate(
        dqc_gateset
    ).verify(circ)


#: Predicate for checking gateset is valid
dqc_gateset_predicate = UserDefinedPredicate(check_function)


def tk2_zzphase_to_crz(a, b, c) -> Circuit:
    """Given a TK2 gate XXPhase(a)*YYPhase(b)*ZZPhase(c), assume that
    a=0 and b=0 so that the gate is simply a ZZPhase(c); then, return
    an equivalent circuit using CRz and Rz gates.

    Note: Unfortunately, pytket does not currently support a simple
    interface to write rebase passes other than those based on replacing
    TK2 gates and CX gates; in this case, we are using the former.
    """
    return Circuit(2).CRz(-2*c, 0, 1).Rz(c, 1)


def tk1_to_Rz_H(a, b, c) -> Circuit:
    """Given a TK1 gate Rz(a)*Rx(b)*Rz(c), return an equivalent circuit
    using Rz and H gates.
    """
    return Circuit(1).Rz(c, 0).H(0).Rz(b, 0).H(0).Rz(a, 0)


#: Pass rebasing gates to those valid within pytket-dqc
dqc_rebase = RebaseCustom(dqc_gateset, tk2_zzphase_to_crz, tk1_to_Rz_H)

#: Defining StartingProcess and EndingProcess custom gates
def_circ = Circuit(2)
def_circ.add_barrier([0, 1])

start_proc = CustomGateDef.define("StartingProcess", def_circ, [])
end_proc = CustomGateDef.define("EndingProcess", def_circ, [])
telep_proc = CustomGateDef.define("Teleportation", def_circ, [])
