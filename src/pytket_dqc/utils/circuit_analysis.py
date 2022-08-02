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
