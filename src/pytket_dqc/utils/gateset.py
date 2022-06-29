from pytket.predicates import (  # type: ignore
    GateSetPredicate,
    NoSymbolsPredicate,
    UserDefinedPredicate
)
from pytket import OpType, Circuit
from pytket.passes import (  # type: ignore
    auto_rebase_pass,
)
from pytket.circuit import CustomGateDef

#: Allowed gateset for distributors in pytket-dqc
dqc_gateset = {OpType.Rx, OpType.CZ, OpType.Rz, OpType.CX,
               OpType.Measure, OpType.CRz}


def check_function(circ):

    return (
        NoSymbolsPredicate().verify(circ) and
        GateSetPredicate(dqc_gateset).verify(circ)
    )


#: Predicate for checking gateset is valid
dqc_gateset_predicate = UserDefinedPredicate(check_function)

#: Pass rebasing gates to those valid within pytket-dqc
dqc_rebase = auto_rebase_pass(dqc_gateset)

#: Defining StartingProcess and EndingProcess custom gates
def_circ = Circuit(2)
def_circ.add_barrier([0, 1])

start_proc = CustomGateDef.define("StartingProcess", def_circ, [])
end_proc = CustomGateDef.define("EndingProcess", def_circ, [])
telep_proc = CustomGateDef.define("Teleportation", def_circ, [])
