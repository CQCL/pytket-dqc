# Copyright 2023 Quantinuum and The University of Tokyo
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pytket import OpType, Circuit


def to_qasm_str(circ: Circuit):
    """Return the QASM representation of the circuit, where the starting
    and ending processes are represented as custom gates.

    :param circ: The circuit in pytket format.
    :type circ: Circuit

    :return: The QASM output as a string.
    :rtype: str
    """

    qasm_str = "OPENQASM 2.0;"
    qasm_str += '\ninclude "qelib1.inc";'
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
        if command.op.get_name() == "starting_process":
            qasm_str += f"\nstarting_process {command.args[0]},{command.args[1]};"
        elif command.op.get_name() == "ending_process":
            qasm_str += f"\nending_process {command.args[0]},{command.args[1]};"
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
            qasm_str += ";"

    return qasm_str + "\n"
