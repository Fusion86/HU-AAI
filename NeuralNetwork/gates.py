from neuron import Perceptron, InputNode


def create_or(inputs):
    return Perceptron(inputs, [1] * len(inputs), -1)


def create_nor(inputs):
    return Perceptron(inputs, [-1] * len(inputs), 1)


def create_and(inputs):
    return Perceptron(inputs, [1] * len(inputs), len(inputs) * -1)


def create_nand(inputs):
    return Perceptron(inputs, [-1] * len(inputs), len(inputs) - 1)


def create_xor(inputs):
    or_gate = create_or(inputs)
    nand_gate = create_nand(inputs)
    return create_and([or_gate, nand_gate])


def gate_tester(name, gate_func, inputs, expected_output):
    input_str = ", ".join([str(x.output) for x in inputs])

    gate = gate_func(inputs)
    print(
        "{:<8} {:<12} = {:<12}".format(name, input_str, gate.output),
        end="",
    )

    if gate.output == expected_output:
        print("(OK)")
    else:
        print("(FAILED)")


if __name__ == "__main__":
    # OR gate
    inputs = [
        InputNode(0),
        InputNode(0),
        InputNode(0),
    ]

    gate_tester("OR", create_or, inputs, 0)

    inputs = [
        InputNode(0),
        InputNode(1),
        InputNode(0),
    ]

    gate_tester("OR", create_or, inputs, 1)

    inputs = [
        InputNode(1),
        InputNode(1),
        InputNode(1),
        InputNode(1),
    ]

    gate_tester("OR", create_or, inputs, 1)

    # NOR gate
    inputs = [
        InputNode(0),
        InputNode(0),
        InputNode(0),
    ]

    gate_tester("NOR", create_nor, inputs, 1)

    # AND gate
    inputs = [
        InputNode(0),
        InputNode(1),
    ]

    gate_tester("AND", create_and, inputs, 0)

    inputs = [
        InputNode(0),
        InputNode(1),
        InputNode(1),
    ]

    gate_tester("AND", create_and, inputs, 0)

    inputs = [
        InputNode(1),
        InputNode(1),
        InputNode(1),
    ]

    gate_tester("AND", create_and, inputs, 1)

    # NAND gate
    inputs = [
        InputNode(0),
        InputNode(1),
    ]

    gate_tester("NAND", create_nand, inputs, 1)

    inputs = [
        InputNode(0),
        InputNode(1),
        InputNode(1),
    ]

    gate_tester("NAND", create_nand, inputs, 1)

    inputs = [
        InputNode(1),
        InputNode(1),
        InputNode(1),
    ]

    gate_tester("NAND", create_nand, inputs, 0)

    # XOR gate
    inputs = [
        InputNode(0),
        InputNode(1),
    ]

    gate_tester("XOR", create_xor, inputs, 1)

    inputs = [
        InputNode(1),
        InputNode(1),
    ]

    gate_tester("XOR", create_xor, inputs, 0)

    inputs = [
        InputNode(0),
        InputNode(0),
    ]

    gate_tester("XOR", create_xor, inputs, 0)

    inputs = [
        InputNode(1),
        InputNode(0),
    ]

    gate_tester("XOR", create_xor, inputs, 1)
