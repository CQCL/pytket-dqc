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


# TODO: This is checked by parsing the name of the qubit.
# Is there a better way to do this?
def is_link_qubit(qubit) -> bool:
    qubit_name = str(qubit).split()
    # ``qubit_name`` either follows either of these patterns:
    #     Workspace qubit: ['Server', server_id+'['+qubit_id+']']
    #     Link qubit: ['Server', server_id, 'Link', 'Edge', edge_id+'[0]']
    return len(qubit_name) > 2
