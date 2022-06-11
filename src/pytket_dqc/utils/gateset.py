from pytket.predicates import (  # type: ignore
    GateSetPredicate,
    NoSymbolsPredicate,
    UserDefinedPredicate
)
from pytket import OpType
from pytket.passes import (  # type: ignore
    auto_rebase_pass,
)

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
