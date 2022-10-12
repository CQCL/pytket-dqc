from pytket import OpType

def to_qasm_str(circ):
    
    qasm_str = "OPENQASM 2.0;"
    qasm_str += "\ninclude \"qelib1.inc\";"
    qasm_str += "\n"
    qasm_str += "\ngate starting_process q,e"
    qasm_str += "\n{"
    qasm_str += "\n\tbarrier q,e;"
    qasm_str += "\n}"
    qasm_str += "\ngate ending_process e,q"
    qasm_str += "\n{"
    qasm_str += "\n\tbarrier e,q;"
    qasm_str += "\n}"
    qasm_str += "\n"

    for register in circ.q_registers:
        qasm_str += f"\nqreg {register.name}[{register.size}];"
    qasm_str += "\n"
    
    for command in circ.get_commands():
        if command.op.get_name() == 'starting_process':
            qasm_str += f'\nstarting_process {command.args[0]},{command.args[1]};'
        elif command.op.get_name() == 'ending_process':
            qasm_str += f'\nending_process {command.args[0]},{command.args[1]};'
        else:
            qasm_str += "\n"
            qasm_str += command.op.type.name.lower()
                                    
            if command.op.type in [OpType.Rz, OpType.CU1]:
                qasm_str += "("
                for param in command.op.params:
                    qasm_str += f"{param}*pi"
                qasm_str += ")"
            
            qasm_str += f" {command.args[0]}"
            for arg in command.args[1:]:
                qasm_str += f",{arg}"
            qasm_str += ';'
    
    return qasm_str + '\n'