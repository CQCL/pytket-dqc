from pytket import Circuit, OpType  # type: ignore
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pytket_dqc import Distribution


class ConstraintException(Exception):
    """Raised when the communication memory constraint of a server is
    exceeded. Stores the offending server and gate vertex at which
    the constraint was violated.
    """
    def __init__(self, message, server):
        super().__init__(message)
        self.server = server
        self.v_gate = None


def all_cu1_local(circ: Circuit) -> bool:
    """Checks that all of the CU1 gates in the circuit are local.
    """
    cu1_gates = [
        gate for gate in circ.get_commands() if gate.op.type == OpType.CU1
    ]
    for gate in cu1_gates:
        q0 = gate.qubits[0]
        q1 = gate.qubits[1]
        if get_server_id(q0) != get_server_id(q1):
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


def detached_gate_count(distribution: 'Distribution') -> int:
    """Scan the distribution and return the number of detached gates in it.
    An detached gate is a 2-qubit gate that acts on link qubits on both
    ends; i.e. it is implemented away from both of its home servers.

    :param distribution: The distribution to be analysed
    :type distribution: Distribution

    :return: The number of detached gates
    :rtype: int
    """

    n_detached = 0

    for vertex in distribution.circuit.vertex_list:

        if not distribution.circuit.is_qubit_vertex(vertex):

            # Qubits gate acts on in original circuit
            q_1, q_2 = distribution.circuit.get_gate_of_vertex(vertex).qubits

            # Vertices of these qubits
            v_1 = distribution.circuit.get_vertex_of_qubit(q_1)
            v_2 = distribution.circuit.get_vertex_of_qubit(q_2)

            # Servers to which the qubits have been assigned
            s_1 = distribution.placement.placement[v_1]
            s_2 = distribution.placement.placement[v_2]

            # Server to which the gate has been assigned
            s_g = distribution.placement.placement[vertex]

            # Count if detached
            if not ((s_1 == s_g) or (s_2 == s_g)):
                n_detached += 1

    return n_detached


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

        if command.op.get_name() == "starting_process":
            starting_count += 1
        elif command.op.get_name() == "ending_process":
            ending_count += 1
        elif command.op.get_name() == "teleportation":
            telep_count += 1

    assert starting_count == ending_count

    return starting_count + telep_count
