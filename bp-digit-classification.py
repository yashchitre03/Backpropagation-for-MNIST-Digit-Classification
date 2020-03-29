import struct as st
import numpy as np
import matplotlib.pyplot as plt


def read_input(file):
    """

    Args:
        file (idx): binary input file.

    Returns:
        numpy: arrays for our dataset.

    """
    
    with open(file, 'rb') as file:
        z, d_type, d = st.unpack('>HBB', file.read(4))
        shape = tuple(st.unpack('>I', file.read(4))[0] for d in range(d))
        return np.frombuffer(file.read(), dtype=np.uint8).reshape(shape)


train_X = read_input('train-images.idx3-ubyte')
train_y = read_input('train-labels.idx1-ubyte')
test_X = read_input('t10k-images.idx3-ubyte')
test_y = read_input('t10k-labels.idx1-ubyte')

input_size = 784
hidden_neurons = 30
output_neurons = 10
w_hidden = np.random.uniform(low=0, high=1, size=(hidden_neurons, input_size))
bias_hidden = np.random.uniform(low=0, high=1, size=(hidden_neurons))
w_output = np.random.uniform(low=0, high=1, size=(output_neurons, hidden_neurons))
bias_output = np.random.uniform(low=0, high=1, size=(output_neurons))


def tanh_act_fn(induced_local_field):
    return np.tanh(induced_local_field)


def sigmoid_act_fn(induced_local_field):
    return 1 / (1 + np.exp(-induced_local_field))


def der_tanh(induced_local_field):
    return 1-(np.tanh(induced_local_field)**2)


def der_sigmoid(induced_local_field):
    sigmoid_u = sigmoid_act_fn(induced_local_field)
    return sigmoid_u * (1 - sigmoid_u)
    
    