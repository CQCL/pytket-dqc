from pytket import Circuit, OpType  # type: ignore
from pytket_dqc.utils.gateset import is_start_proc, is_end_proc, is_telep_proc


def all_gates_local(circ: Circuit) -> bool:
    """Checks that all of the gates in the circuit are local.
    """
    for cmd in circ.get_commands():
        if cmd.op.type in [OpType.CustomGate, OpType.Barrier]:
            continue
        server = get_server_id(cmd.qubits[0])
        if any(get_server_id(q) != server for q in cmd.qubits):
            return False
    return True


def ebit_memory_required(circ: Circuit) -> dict[int, int]:
    """Scan the circuit and find, for each server, the maximum number of
    ebits simultaneously linked to it. This corresponds to the minimum
    memory dedicated to ebits this circuit requires.

    :param circ: The circuit to be analysed
    :type circ: Circuit

    :return: A dictionary mapping servers to ebit memory required. In
        particular, dict[2] == 3 means that, at some point in the circuit,
        there are 3 ebits simultaneously sharing a qubit with server 2.
    :rtype: dict[int, int]
    """
    ebit_memory_required = dict()
    current = dict()

    # Find all server IDs and initialise their ebit memory to 0
    for qubit in circ.qubits:
        server_id = get_server_id(qubit)
        ebit_memory_required[server_id] = 0
        current[server_id] = 0

    # Scan the circuit for starting and ending EJPP processes and update
    #   the ebit memory requirement accordingly
    for command in circ.get_commands():

        # Increase the current memory if an EJPP process starts
        if command.op.get_name() == "starting_process":
            link_qubit = command.qubits[1]
            assert is_link_qubit(link_qubit)
            server_id = get_server_id(link_qubit)
            current[server_id] += 1

            # Check if the ebit memory needs to be increased
            if current[server_id] > ebit_memory_required[server_id]:
                ebit_memory_required[server_id] += 1

        # Decrease the current memory if an EJPP process ends
        elif command.op.get_name() == "ending_process":
            link_qubit = command.qubits[0]
            assert is_link_qubit(link_qubit)
            server_id = get_server_id(link_qubit)
            current[server_id] -= 1

    return ebit_memory_required


def evicted_gate_count(circ: Circuit) -> int:
    """Scan the circuit and return the number of evicted gates in it.
    An evicted gate is a 2-qubit gate that acts on link qubits on both
    ends; i.e. it is implemented away from both of its home servers.

    :param circ: The circuit to be analysed
    :type circ: Circuit

    :return: The number of evicted gates
    :rtype: dict[int, int]
    """
    n_evicted = 0

    for command in circ.get_commands():
        if command.op.type in {OpType.CU1, OpType.CZ}:
            qubits = command.qubits
            if is_link_qubit(qubits[0]) and is_link_qubit(qubits[1]):
                n_evicted += 1

    return n_evicted


# TODO: This is checked by parsing the name of the qubit.
# Is there a better way to do this?
def is_link_qubit(qubit) -> bool:
    qubit_name = str(qubit).split("_")
    # ``qubit_name`` follows either of these patterns:
    #     Workspace qubit: ['server', server_id+'['+qubit_id+']']
    #     Link qubit: ['server', server_id, 'link', 'register['+qubit_id+']']
    # Sanity check
    return len(qubit_name) > 2


# TODO: The way the server ID is obtained is by parsing the name of
# the qubit. Is there a better way to access this information?
def get_server_id(qubit) -> int:
    qubit_name = str(qubit).split("_")
    # ``qubit_name`` follows either of these patterns:
    #     Workspace qubit: ['server', server_id+'['+qubit_id+']']
    #     Link qubit: ['server', server_id, 'link', 'register['+qubit_id+']']
    # Sanity check
    assert qubit_name[0] == "server"

    if is_link_qubit(qubit):
        return int(qubit_name[1])
    else:
        return int(qubit_name[1].split("[")[0])


# This function is for sanity checking and should not make it into production
# (whatever that means for this project). In particular it is not
# documented or tested.
def _cost_from_circuit(circ: Circuit) -> int:

    starting_count = 0
    ending_count = 0
    telep_count = 0

    for command in circ.get_commands():

        if is_start_proc(command):
            starting_count += 1
        elif is_end_proc(command):
            ending_count += 1
        elif is_telep_proc(command):
            telep_count += 1

    assert starting_count == ending_count, f"{starting_count}, {ending_count}"

    return starting_count + telep_count
