import struct

with open("model/sample.emodel", "wb") as f:
    # Header
    f.write(b'EMDL')              # Magic
    f.write(struct.pack('B', 1))  # Version
    f.write(struct.pack('B', 2))  # Layer count (Dense + ReLU)

    # Layer 1: Dense (input=3, output=2)
    f.write(struct.pack('B', 1))         # Layer type = Dense
    f.write(struct.pack('<H', 3))        # input_dim = 3
    f.write(struct.pack('<H', 2))        # output_dim = 2
    f.write(struct.pack('<H', 6))        # weight_count = 3x2
    f.write(struct.pack('<H', 2))        # bias_count = 2

    weights = [0.1, 0.2,   # for neuron 1
               0.3, 0.4,   # for neuron 2
               0.5, 0.6]   # for neuron 3
    biases = [0.5, -0.5]

    for w in weights:
        f.write(struct.pack('<f', w))
    for b in biases:
        f.write(struct.pack('<f', b))

    # Layer 2: ReLU
    f.write(struct.pack('B', 2))         # Layer type = ReLU
    f.write(struct.pack('<H', 2))        # input_dim
    f.write(struct.pack('<H', 2))        # output_dim
    f.write(struct.pack('<H', 0))        # weight_count
    f.write(struct.pack('<H', 0))        # bias_count
