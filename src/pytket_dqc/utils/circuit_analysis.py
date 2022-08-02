from pytket import Circuit  # type: ignore


# This function is for sanity checking and should not make it into production
# (whatever that means for this project). In particular it is not
# documented or tested.
def _cost_from_circuit(circ: Circuit) -> int:

    starting_count = 0
    ending_count = 0
    telep_count = 0

    for command in circ.get_commands():

        if command.op.get_name() == 'StartingProcess':
            starting_count += 1
        elif command.op.get_name() == 'EndingProcess':
            ending_count += 1
        elif command.op.get_name() == 'Teleportation':
            telep_count += 1

    assert starting_count == ending_count

    return starting_count + telep_count

def circuits_are_equivalent(circuit1, circuit2):
    """Given two circuits, compare whether the commands,
    command qubits and quantum registers are equal.

    Essentially compares two circuits and checks if they are equivalent.

    :param circuit1: The first circuit to compare.
    :type circuit1: pytket.circuit.Circuit
    :param circuit2: The second circuit to compare.
    :type circuit2: pytket.circuit.Circuit
    :return: Whether the two circuits are equivalent.
    :rtype: bool
    """

    circuit1_command_names = [command.op.get_name()
                               for command in circuit1.get_commands()]
    circuit2_command_names = [
        command.op.get_name() for command in circuit2.get_commands()]

    circuit1_command_qubits = [
        command.qubits for command in circuit1.get_commands()]
    circuit2_command_qubits = [
        command.qubits for command in circuit2.get_commands()]

    return (
        circuit1_command_names == circuit2_command_names
        and circuit1_command_qubits == circuit2_command_qubits
        and circuit1.q_registers == circuit2.q_registers
    )
