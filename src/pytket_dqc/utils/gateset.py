from pytket.predicates import GateSetPredicate  # type: ignore
from pytket import OpType
from pytket.passes import (  # type: ignore
    auto_rebase_pass,
)

#: Allowed gateset for distributors in pytket-dqc
dqc_gateset = {OpType.Rx, OpType.CZ, OpType.Rz, OpType.CX,
               OpType.Measure, OpType.QControlBox}

#: Predicate for checking gateset is valid
dqc_gateset_predicate = GateSetPredicate(dqc_gateset)

#: Pass rebasing gates to those valid within pytket-dqc
dqc_rebase = auto_rebase_pass(dqc_gateset)