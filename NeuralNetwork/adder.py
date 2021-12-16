from neuron import InputNode
from gates import create_xor, create_and, create_or


def create_adder(a: InputNode, b: InputNode, carry: InputNode):
    xor_1 = create_xor([a, b])
    xor_2 = create_xor([xor_1, carry])

    and_1 = create_and([xor_1, carry])
    and_2 = create_and([a, b])
    or_gate = create_or([and_1, and_2])

    # (Sum, Carry)
    return (xor_2, or_gate)


if __name__ == "__main__":
    adder = create_adder(InputNode(1), InputNode(1), InputNode(1))
    print("Sum: {}, Carry: {}".format(adder[0].output, adder[1].output))
