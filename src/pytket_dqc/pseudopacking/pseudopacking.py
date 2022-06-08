from pytket_dqc.circuits import DistributedCircuit
from pytket import Circuit, OpType
from pytket.circuit import Op, Qubit, CustomGateDef # type: ignore
from pytket.circuit.display import render_circuit_jupyter
from pytket_dqc.placement import Placement
import json
import numpy as np
from typing import cast
from warnings import warn

# The following code is duplicated from distributed_circuit.py
##########################################################################################
def_circ = Circuit(2)
def_circ.add_barrier([0, 1])

sp = 'StartingProcess'
ep = 'EndingProcess'

start_proc = CustomGateDef.define(sp, def_circ, [])
end_proc = CustomGateDef.define(ep, def_circ, [])

def get_circuit_qubit_to_server_qubit_map(dist_circ, placement):
    # Given a placement, return a map that maps circuit qubits -> server qubits
    server_to_qubit_vertex_list = {
        server: [
            vertex for vertex in dist_circ.get_qubit_vertices()
            if placement.placement[vertex] == server
        ]
        for server in set(placement.placement.values())
    }

    # New circuit including distribution gates
    circ = Circuit()

    # A dictionary mapping servers to the qubit registers they contain.
    server_to_register = {}

    # A dictionary mapping (server, hyperedge) pairs to link qubits.
    # The first index is the server the link qubit is contained in.
    # The second index is the hyperedge the link qubit is consumed by.
    server_to_link_register: dict[int, QubitRegister] = {}

    # Here a register is defined for each server. It contains a number of
    # qubits equal to the number of qubit vertices which are placed in the
    # server, +1 qubit for each hyperedge which requires a qubit be
    # moved to this server. Note that this means there is an additional
    # link qubit per e-bit. By the completion of this loop the circuit
    # has all the qubits necessary to complete the computaitons.
    for server, qubit_vertex_list in server_to_qubit_vertex_list.items():

        # Add a register for all of the qubits assigned to this server.
        server_to_register[server] = circ.add_q_register(
            f'Server {server}', len(qubit_vertex_list))

        server_to_link_register[server] = {}
        # For each hyperedge, add the necessary link qubits
        for index, hyperedge in enumerate(dist_circ.hyperedge_list):

            # List of gate vertices in this hyperedge
            gate_vertex_list = [
                vertex for vertex in cast(list[int], hyperedge['hyperedge'])
                if dist_circ.vertex_circuit_map[vertex]['type'] == 'gate'
            ]

            # Find the one unique vertex in the hyperedge which
            # corresponds to a qubit.
            hyperedge_qubit_vertex_list = [
                vertex for vertex in cast(list[int], hyperedge['hyperedge'])
                if vertex not in gate_vertex_list
            ]
            assert len(hyperedge_qubit_vertex_list) == 1
            hyperedge_qubit_vertex = hyperedge_qubit_vertex_list[0]

            # Add a link qubits if the qubit of the hyperedge is not
            # placed in this server, but there is a gate vertex in this
            # hyperedge which is placed in this server.
            if not (placement.placement[hyperedge_qubit_vertex] == server):

                unique_server_used = set([
                    placement.placement[gate_vertex]
                    for gate_vertex in gate_vertex_list
                ])
                if server in unique_server_used:
                    register = circ.add_q_register(
                        f'Server {server} Link Edge {index}', 1)
                    server_to_link_register[server][index] = register[0]

    # Dictionary mapping qubit vertices in hypergraph to server qubit
    # registers
    qubit_vertex_to_server_qubit = {}
    for server, register in server_to_register.items():
        for i, vertex in enumerate(server_to_qubit_vertex_list[server]):
            qubit_vertex_to_server_qubit[vertex] = register[i]

    # Dictionary mapping qubits in original circuit to
    # qubits in new registers
    circuit_qubit_to_server_qubit = {
        dist_circ.vertex_circuit_map[qubit_vertex]['node']: qubit
        for qubit_vertex, qubit in qubit_vertex_to_server_qubit.items()
    }
    return (circuit_qubit_to_server_qubit)
##########################################################################################

def build_qubit_op_list(circuit):
    # Given a circuit, contruct a dictionary from qubits -> [all ops that act on the qubit] where the op is stored as an OpDict.
    op_list = {}
    for qubit in circuit.qubits:
        op_list[qubit] = [] # The list of OpDicts acting on the qubit.

    for i, command in enumerate(circuit.get_commands()):
        for qubit in command.qubits:
            op_dict = command_to_op_dict(command)
            op_list[qubit].append(op_dict)
    return op_list

def is_diagonal(op):
    #Boolean function that determines if a given command has an associated matrix representation (in the computational basis) that is diagonal.
    #This function uses the fastest answer presented here https://stackoverflow.com/questions/43884189/check-if-a-large-matrix-is-diagonal-matrix-in-python
    # TODO: Test for symbolic Rx gates

    diagonal_ops = [OpType.Rz, OpType.CZ]
    if op.type in diagonal_ops: # In this way we can handle operations with symbolic parameters.
        return True
    try: # Assume if there's an error that it is not a diagonal gate (error is probably because symbolic).
        array = op.get_unitary().round(12) # To stay consistent with TKET team.
        i, j = array.shape
        test = array.reshape(-1)[:-1].reshape(i-1, j+1)
    except:
        return False
    return ~np.any(test[:, 1:])


def is_antidiagonal(op):
    # See is_diagonal() for discussion
    try:
        array = np.flip(op.get_unitary(), 0).round(12)
        i, j = array.shape
        test = array.reshape(-1)[:-1].reshape(i-1, j+1)
    except:
        return False
    return ~np.any(test[:, 1:])


def remove_packables(circuit):
    # Contruct a new circuit from the input circuit, with all the packable 1 qubit gates removed
    new_circ = Circuit(len(circuit.qubits))
    new_circ_commands = []

    for command in circuit.get_commands():
        if len(command.qubits) > 1:
            new_circ_commands.append(command)
        elif not (is_antidiagonal(command.op) or is_diagonal(command.op)):
            new_circ_commands.append(command)
    
    for command in new_circ_commands:
        op_dict = command_to_op_dict(command)
        add_op_dict_to_circuit(new_circ, op_dict)

    return new_circ

def command_to_op_dict(command):
    # Converts a circuit.Command to an OpDict.

    op_dict = OpDict(command.op, command.args)
    return op_dict

def add_op_dict_to_circuit(circuit, op_dict):
    # Given an OpDict and a circuit, add the OpDict to that circuit.
    if op_dict.op.get_name() == 'StartingProcess':
        circuit.add_custom_gate(start_proc, [], op_dict.args)
    elif op_dict.op.get_name() == 'EndingProcess':
        circuit.add_custom_gate(end_proc, [], op_dict.args)
    else:
        circuit.add_gate(op_dict.op, op_dict.args)

def copy_op(op):
    # Returns a copy of a circuit.Op.
    # This doesn't seem like a terribly efficient way to create copies of SPs and EPs though
    if op.type != OpType.CustomGate:
        return Op.create(op.type, op.params)
    else: # Can the following be improved upon?
        circ = Circuit(2)
        if op.get_name() == sp:
            circ.add_custom_gate(start_proc, [], circ.qubits)
        else:
            circ.add_custom_gate(end_proc, [], circ.qubits)
    return circ.get_commands()[0].op

def copy_qubit(qubit):
    # Copies a Qubit.
    copy = Qubit(name = qubit.reg_name, index = qubit.index)
    return copy

def copy_qubit_ops_dict(qubit_ops_dict):
    # Given a dictionary mapping qubit -> [all OpDicts acting on that qubit] return a copy of the dictionary.
    copy_ops_dict = {}
    for qubit, op_dict_list in qubit_ops_dict.items():
        copy_ops_dict[qubit] = []
        for op_dict in op_dict_list:
            copy_ops_dict[qubit].append(op_dict.copy())
    return copy_ops_dict

def build_l_qubit_to_s_qubit_map(dc_ops):
    # Given a dictionary mapping qubit -> [all OpDicts acting on that qubit], return a dictionary mapping link qubit -> connected server qubit.
    l_qubit_to_s_qubit_map = {}
    for l_qubit in [qubit for qubit in dc_ops.keys() if is_l_qubit(qubit)]:
        for op_dict in dc_ops[l_qubit]:
            if op_dict.op.get_name() == 'StartingProcess':
                l_qubit_to_s_qubit_map[l_qubit] = op_dict.args[0] #the s_qubit is always the 0th entry in the args

    # Sometimes we might have to link server0 -> server2 via server1, in which case the 's_qubit' is still an l_qubit
    # however we want to find the true s_qubit so follow the links until we've done this
    false_s_qubits = []
    for l_qubit, s_qubit in l_qubit_to_s_qubit_map.items():
        while s_qubit in l_qubit_to_s_qubit_map.keys():
            s_qubit = l_qubit_to_s_qubit_map[s_qubit]
        l_qubit_to_s_qubit_map[l_qubit] = s_qubit
    return l_qubit_to_s_qubit_map

def is_l_qubit(qubit):
    # Bool evaluation of whether a given qubit is a link qubit or not.
    return len(qubit.reg_name.split(' ')) > 2

def reinsert_cleaned_ops(og_ops, dc_ops, c_qubit_to_s_qubit_map, debug = False):
    # Reinsert the removed (anti)diagonal gates, with correctional X gates where needed
    # o(ri)g(inal circuit)_ops - all the ops on the original circuit
    # d(istributed)c(ircuit)_ops - all the ops on the distributed circuit
    # c(ircuit)_qubit_to_s(erver)_qubit_map - a map from the circuit qubits to the server qubit they are distributed onto

    log = [] # (Somewhat) helps with debugging when errors come up.

    # Insert the removed ops into a copy of dc_ops dict    
    new_dc_ops = copy_qubit_ops_dict(dc_ops)
    
    sp = 'StartingProcess'
    ep = 'EndingProcess'

    l_qubit_to_s_qubit = build_l_qubit_to_s_qubit_map(dc_ops) # a dictionary mapping each l_qubit -> s_qubit

    for c_qubit in og_ops.keys():
        s_qubit = c_qubit_to_s_qubit_map[c_qubit]
        qt = QubitTracker(c_qubit, s_qubit)

        # We will iterate over the placed ops on the c_qubit and its associated s_qubit.
        # Possible cases are:
        # 
        # 1) The ops match:
        #   -> Nothing needs be done
        #   -> Increment s_qubit_i
        # 2) The c_qubit op is packable, so was removed in the pre-processing step
        #   -> Place the c_qubit op back on the s_qubit (if antidiagonal then also insert appropriate X corrections)
        #   -> Increment s_qubit_i (Increment l_qubit_i)
        # 3) The placed c_qubit op is global, and this s_qubit has been teleported to a different server 
        #   -> Should be linked to at least one l_qubit in this case
        #       -> Follow the l_qubit
        #       -> Compare with the command at position l_qubit_i with revised args of [l_qubit, other_arg_qubit]
        #   -> If matching, we are done
        #   -> Increment l_qubit_i
        # 4) The placed c_qubit op is global, but it is moved to same server as the s_qubit for this c_qubit
        #   -> Compare the command at position s_qubit_i, but with revised qubit arguments [s_qubit, l_qubit_on_s_qubit_server]
        #   -> If matching, we are done
        #   -> Increment s_qubit_i
        # 5) The placed c_qubit op is local, but there is an SP on the s_qubit
        #   -> Open a l_qubit on s_qubit
        #   -> Increment s_qubit_i
        # 6) The placed c_qubit op is local, but there is an EP on the s_qubit
        #   -> Close the l_qubit related to this EP
        #   -> (Verify all the commands have been executed?)
        #   -> Increment s_qubit_i
        # 7) The placed_op_dict has been teleported to a server separate to both arg qubits
        #   -> Follow onto all relevant l_qubits for s_qubit.
        #   -> Check the current op_dict on each l_qubit and see if both the 
        #   -> Increment l_qubit_i
 
        for og_op_dict in og_ops[c_qubit]:
            placed_op_dict = og_op_dict.copy()
            placed_op_dict.place(c_qubit_to_s_qubit_map) # Place the op_dict
            if len(placed_op_dict.args) > 1:
                other_arg_qubit = [qubit for qubit in placed_op_dict.args if qubit != s_qubit][0]
                other_arg_reg_num = get_qubit_reg_num(other_arg_qubit)
            log.append(f'MATCHING THE FOLLOWING OP: {placed_op_dict}, c_qubit is {c_qubit} s_qubit is {s_qubit}\n')
        
            ops_equal = False
            while not ops_equal: # Keep looping through until the matching operation is found.
                relevant_active_l_qubits = [l_qubit for l_qubit in qt.get_active_l_qubits() if l_qubit_to_s_qubit[l_qubit] == s_qubit]

                dc_op_dict = None
                if qt.get_s_qubit_i() < len(new_dc_ops[s_qubit]):
                    dc_op_dict = new_dc_ops[s_qubit][qt.get_s_qubit_i()]
                    log.append(f'dc_op_dict is {dc_op_dict}\n')

                try:
                    if ( # Case 2
                        placed_op_dict.is_packable()
                        and len(placed_op_dict.args) == 1
                        ):
                        ops_equal = case2(qt, new_dc_ops, placed_op_dict, relevant_active_l_qubits)

                    elif ( # Case 1
                        dc_op_dict != None
                        and placed_op_dict.is_equal(dc_op_dict)
                        ):
                        ops_equal = case1(qt)

                    elif ( # Case 5
                        dc_op_dict != None
                        and dc_op_dict.op.get_name() == sp
                        ):
                        ops_equal = case5(qt, dc_op_dict)

                    elif ( # Case 6
                        dc_op_dict != None
                        and dc_op_dict.op.get_name() == ep
                        ):
                        ops_equal = case6(qt, new_dc_ops, dc_op_dict)

                    elif ( # Case 3
                        not placed_op_dict.is_local()
                        and any(get_qubit_reg_num(l_qubit) == other_arg_reg_num for l_qubit in relevant_active_l_qubits)
                        ):
                        try:
                            ops_equal = case3(qt, new_dc_ops, placed_op_dict, relevant_active_l_qubits, other_arg_qubit, other_arg_reg_num)
                        except:
                            raise ValueError(
                                f'''
                                {[l_qubit for l_qubit in relevant_active_l_qubits if get_qubit_reg_num(l_qubit) == other_arg_reg_num]} is the l_qubit in question.
                                ''')

                    elif ( # Case 4
                        dc_op_dict != None # Must have an op_dict on this s_qubit
                        and not placed_op_dict.is_local() # The placed_op_dict must be across two servers
                        and len(dc_op_dict.args) > 1 # prevents failure of the next test if dc_op_dict happens to be 1 qubit gate
                        and set([l_qubit_to_s_qubit[dc_op_dict.arg_l_qubits()[0]], s_qubit]) == set(placed_op_dict.args) # The arg qubits of the dc_op_dict must be the s_qubit and an l_qubit which links to the other_arg_qubit
                        ):
                        ops_equal = case4(qt, new_dc_ops, placed_op_dict, dc_op_dict, other_arg_qubit, l_qubit_to_s_qubit)

                    elif ( # Case 7
                        not placed_op_dict.is_local()
                        ): #This condition does not verify we have case 7 fully, but if we have captured all the possible cases then we should be ok. Coming up with a full condition which verifies we have case 7 would be nicer though.
                        ops_equal = case7(qt, new_dc_ops, relevant_active_l_qubits, placed_op_dict, l_qubit_to_s_qubit)

                    else:
                        raise ValueError(f'''
                        None of the 7 cases have been triggered.
                        c_qubit: {c_qubit} is placed on s_qubit: {s_qubit}.
                        placed_op_dict: {placed_op_dict}, dc_op_dict: {dc_op_dict}.
                        s_qubit_i: {qt.get_s_qubit_i()}.
                        ''')
                except:
                    log_string = ''.join([line for line in log[-10:]])
                    raise ValueError(
                        f'''An error has occurred.
                        s_qubit is {s_qubit}.
                        Current s_qubit_i is {qt.get_s_qubit_i()}, sp_count is {qt.sp_count}, ep_count is {qt.ep_count}.
                        The op we are trying to match is {placed_op_dict}.
                        Our dc_op_dict (if there is one) is {dc_op_dict}.
                        l_qubit is {dc_op_dict.arg_l_qubits()[0]} which is linked to {l_qubit_to_s_qubit[dc_op_dict.arg_l_qubits()[0]]}.
                        The last 10 lines in the log are:
                        {log_string}
                        ''')

    return new_dc_ops

def case1(qt):
    qt.increment_s_qubit_i()
    return True

def case2(qt, ops_list, placed_op_dict, relevant_active_l_qubits):
    ops_list[qt.get_s_qubit()].insert(qt.get_s_qubit_i(), placed_op_dict)
    qt.increment_s_qubit_i()
    if is_antidiagonal(placed_op_dict.op):
        for l_qubit in relevant_active_l_qubits:
            X_op = Op.create(OpType.X)
            X_op_dict = OpDict(X_op, [l_qubit])
            ops_list[l_qubit].insert(qt.get_l_qubit_i(l_qubit), X_op_dict)
    return True

def case3(qt, ops_list, placed_op_dict, relevant_active_l_qubits, other_arg_qubit, other_arg_reg_num):
    l_qubit = [l_qubit for l_qubit in relevant_active_l_qubits if get_qubit_reg_num(l_qubit) == other_arg_reg_num][0]
    l_qubit_op_dict = ops_list[l_qubit][qt.get_l_qubit_i(l_qubit)]
    placed_op_dict_copy = placed_op_dict.copy()
    placed_op_dict_copy.args = [other_arg_qubit, l_qubit]
    qt.increment_l_qubit_i(l_qubit)

    if ops_list[l_qubit][qt.get_l_qubit_i(l_qubit)].op.get_name() == 'EndingProcess': #if we are now pointing at an ending process 
        qt.end_l_qubit(l_qubit)
    return placed_op_dict_copy.is_equal(l_qubit_op_dict)

def case4(qt, ops_list, placed_op_dict, dc_op_dict, other_arg_qubit, l_qubit_to_s_qubit):
    l_qubit = [l_qubit for l_qubit in dc_op_dict.args if (l_qubit != qt.get_s_qubit()) and (l_qubit_to_s_qubit[l_qubit] == other_arg_qubit)][0]
    placed_op_dict_copy = placed_op_dict.copy()
    placed_op_dict_copy.args = [qt.get_s_qubit(), l_qubit]
    qt.increment_s_qubit_i()
    return placed_op_dict_copy.is_equal(dc_op_dict)

def case5(qt, dc_op_dict):
    l_qubit = [qubit for qubit in dc_op_dict.args if qubit != qt.get_s_qubit()][0]
    qt.start_l_qubit(l_qubit)
    qt.increment_s_qubit_i()
    qt.increment_sp_count()
    return False

def case6(qt, ops_list, dc_op_dict):
    qt.increment_s_qubit_i()
    qt.increment_ep_count()
    return False

def case7(qt, ops_list, relevant_active_l_qubits, placed_op_dict, l_qubit_to_s_qubit):
    for l_qubit in relevant_active_l_qubits:
        current_l_qubit_op_dict = ops_list[l_qubit][qt.get_l_qubit_i(l_qubit)].copy()
        current_l_qubit_op_dict.set_args([
            l_qubit_to_s_qubit[current_l_qubit_op_dict.args[0]],
            l_qubit_to_s_qubit[current_l_qubit_op_dict.args[1]]
        ])

        if current_l_qubit_op_dict.is_equal(placed_op_dict):
            qt.increment_l_qubit_i(l_qubit)

            if ops_list[l_qubit][qt.get_l_qubit_i(l_qubit)].op.get_name() == 'EndingProcess':
                qt.end_l_qubit(l_qubit)
            return True
    
    raise ValueError(f'''
        Assumed that there has been an evicted gate, but it seems that this has not been obtained.
    ''')

def op_list_to_ordered_ops(op_list):
    #convert a dictionary of qubit -> [all the OpDicts which act on the qubit] to an ordered list of op_dicts where each op occurs once
    ordered_ops = []
    ref_op_list = copy_qubit_ops_dict(op_list)
    for qubit, op_dict_list in ref_op_list.items():
        while len(op_dict_list) > 0:
            op_dict = op_dict_list[0]
            if len(op_dict.args) == 1: #then it's a one qubit gate so we can add it now
                ordered_ops.append(op_dict)
                del op_dict_list[0]
            else: #need to add all the ops prior to this global one to the ordered ops list
                append_prior_local_ops(op_dict, qubit, ref_op_list, ordered_ops)
                
    return ordered_ops

def append_prior_local_ops(this_op_dict, this_qubit, op_list, ordered_ops):
    #add local ops to ordered ops prior to the specified op_dict of the global op
    #calling this function again if we find another global op whilst traversing backwards
    copy_op_dict = this_op_dict.copy()
    copy_op_dict.set_args([copy_qubit(qubit_arg) for qubit_arg in this_op_dict.args if qubit_arg != this_qubit])
    connected_qubit = copy_op_dict.args[0]

    while not op_list[connected_qubit][0].is_equal(this_op_dict):
        rel_op_dict = op_list[connected_qubit][0]
        if len(rel_op_dict.args) == 1:
            ordered_ops.append(rel_op_dict)
            del op_list[connected_qubit][0]
        else:
            append_prior_local_ops(rel_op_dict, connected_qubit, op_list, ordered_ops)

    ordered_ops.append(this_op_dict) #now we can append the global op
    del op_list[this_qubit][0] #and now the op can be deleted from both qubits
    del op_list[connected_qubit][0]

    return

def ordered_list_to_circuit(op_list):
    # Given an ordered list of OpDicts, build a circuit.
    circ = Circuit()
    ordered_ops = op_list_to_ordered_ops(op_list)
    q_regs = {}
    for qubit_key in op_list.keys():
        reg_name = qubit_key.reg_name
        if reg_name not in q_regs.keys():
            q_regs[reg_name] = [qubit_key]
        else:
            q_regs[reg_name].append(qubit_key)
    
    for reg_name in q_regs.keys():
        circ.add_q_register(reg_name, len(q_regs[reg_name]))
    
    for op_dict in ordered_ops:
        add_op_dict_to_circuit(circ, op_dict)
    return circ

def get_qubit_reg_num(qubit):
    # Return the register number of the given qubit.
    reg_no = qubit.reg_name.split(' ')[1]
    return int(reg_no)

class QubitTracker():
    def __init__(self, c_qubit, s_qubit):
        self.c_qubit = c_qubit
        self.s_qubit = s_qubit
        self.s_qubit_i = 0
        self.l_qubits = {}
        self.sp_count = 0
        self.ep_count = 0
            
    def get_s_qubit_i(self):
        return self.s_qubit_i

    def get_s_qubit(self):
        return self.s_qubit

    def increment_s_qubit_i(self):
        self.s_qubit_i += 1

    def increment_l_qubit_i(self, l_qubit):
        self.l_qubits[l_qubit]['i'] += 1

    def get_l_qubit_i(self, l_qubit):
        return self.l_qubits[l_qubit]['i']

    def start_l_qubit(self, l_qubit):
        self.l_qubits[l_qubit] = {
            'i': 1,
            'l_qubit': l_qubit,
            'is_active': True
        }

    def end_l_qubit(self, l_qubit):
        self.l_qubits[l_qubit]['is_active'] = False
            
    def get_linked_status(self):
        linked = False
        for l_qubit_dict in self.l_qubits.values():
            linked = linked or l_qubit_dict['is_active']
        return linked

    def get_active_l_qubits(self):
        l_qubits = [l_qubit_dict['l_qubit'] for l_qubit_dict in self.l_qubits.values() if l_qubit_dict['is_active']]
        return l_qubits

    def get_all_l_qubits(self):
        l_qubits = [l_qubit for l_qubit in self.l_qubits.keys()] # might be ok to just return self.l_qubit.keys() but I'm not sure regarding copies
        return l_qubits

    def increment_sp_count(self):
        self.sp_count += 1
    
    def increment_ep_count(self):
        self.ep_count += 1

class OpDict():
    def __init__(self, op, args):
        self.op = op #the operation in question
        self.args = args #the qubit we are interested in

    def copy(self):
        args_copy = []
        for qubit in self.args:
            args_copy.append(copy_qubit(qubit))
        copy = OpDict(copy_op(self.op), args_copy)
        return copy

    def set_args(self, args):
        self.args = args

    def is_equal(self, op_dict):
        if not set(self.args) == set(op_dict.args):
            return False
        elif self.op.type == OpType.CustomGate and op_dict.op.type == OpType.CustomGate:
            return self.op.get_name() == op_dict.op.get_name()
        return self.op.type == op_dict.op.type and self.op.params == op_dict.op.params

    def place(self, c_qubit_to_s_qubit_map):
        assert not self.is_placed(), 'This OpDict is already placed!'
        new_arg_list = []

        for c_qubit in self.args:
            new_arg_list.append(c_qubit_to_s_qubit_map[c_qubit])

        self.set_args(new_arg_list)

    def is_placed(self):
        return self.args[0].reg_name != 'q'

    def is_local(self):
        assert self.is_placed(), 'This OpDict has not been placed on a server qubit(s).'
        if len(self.args) == 1:
            return True
        reg_num = get_qubit_reg_num(self.args[0])

        for s_qubit in self.args[1:]:
            if get_qubit_reg_num(s_qubit) != reg_num:
                return False
        return True

    def is_packable(self):
        return is_antidiagonal(self.op) or is_diagonal(self.op)

    def get_n_qubits(self):
        return len(self.args)

    def __repr__(self):
        string = f'Op: {self.op.get_name()} Args: {self.args}'
        return string

    def arg_l_qubits(self):
        l_qubits = []
        for qubit in self.args:
            if len(qubit.reg_name.split(' ')) > 2:
                l_qubits.append(qubit)
        return l_qubits
