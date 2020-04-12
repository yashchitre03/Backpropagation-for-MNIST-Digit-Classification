import struct as st
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

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


train_size = 60000
test_size = 10000
input_size = 784
hidden_neurons = 75
output_neurons = 10

train_X = read_input('train-images.idx3-ubyte')
train_y = read_input('train-labels.idx1-ubyte')
test_X = read_input('t10k-images.idx3-ubyte')
test_y = read_input('t10k-labels.idx1-ubyte')

train_mean = np.mean(train_X[0:train_size].flatten())

w_hidden = np.random.normal(loc=0, scale=1, size=(hidden_neurons, input_size))
bias_hidden = np.random.normal(loc=1, scale=1, size=(hidden_neurons))
w_output = np.random.normal(loc=0, scale=1, size=(output_neurons, hidden_neurons))
bias_output = np.random.normal(loc=1, scale=1, size=(output_neurons))


def tanh_act_fn(induced_local_field):
    return np.tanh(induced_local_field)


def sigmoid_act_fn(induced_local_field):
    return 1 / (1 + np.exp(-1*induced_local_field))


def der_tanh(induced_local_field):
    return 1 - (np.tanh(induced_local_field)**2)


def der_sigmoid(induced_local_field):
    sigmoid_u = sigmoid_act_fn(induced_local_field)
    return sigmoid_u * (1 - sigmoid_u)
    
    
eta = 0.01
epoch = 0
train_energy_list = []
train_error_list = []
test_energy_list = []
test_error_list = []

while True:
    train_error = train_energy = 0
    for i in range(train_size):
        
        x = (train_X[i].flatten() - train_mean) / 255
        
        # feedforward
        V = w_hidden @ x + bias_hidden
        y = tanh_act_fn(V)
        V_d = w_output @ y + bias_output
        y_d = sigmoid_act_fn(V_d)
        
        # one hot encoding
        actual_one_hot = np.empty((10, 1))
        desired_one_hot = np.empty((10, 1))
        actual_one_hot.fill(0.1)
        desired_one_hot.fill(0.1)
        actual_one_hot[np.argmax(y_d), :] = 0.9
        desired_one_hot[train_y[i], :] = 0.9
        
        # feedback
        e = (desired_one_hot - y_d.reshape((10,1))).reshape(10) 
        last_layer_bp = der_sigmoid(V_d) * e
        first_layer_bp = der_tanh(V) * (w_output.T @ last_layer_bp)
        
        # weight gradient
        delta_w_hidden = (-1 * x.reshape(784,1) @ first_layer_bp.reshape(hidden_neurons, 1).T).T
        delta_bias_hidden = -1 * first_layer_bp
        delta_w_output = (-1 * y.reshape(hidden_neurons,1) @ last_layer_bp.reshape(1,10)).T
        delta_bias_output = -1 * last_layer_bp
        
        # weight update
        w_hidden = w_hidden - eta * delta_w_hidden
        bias_hidden = bias_hidden - eta * delta_bias_hidden
        w_output = w_output - eta * delta_w_output
        bias_output = bias_output - eta * delta_bias_output
        
        train_energy += np.linalg.norm(e)**2
        if not np.array_equal(desired_one_hot, actual_one_hot):
            train_error += 1
            
    train_energy /= train_size
    
    train_energy_list.append(train_energy)
    train_error_list.append(train_error)
    print(f"eta: {eta:.2f}\tepoch: {epoch:5} ->")
    print(f"Train Error: {train_error/train_size:.4f}")
    
    # test accuracy
    test_error = test_energy = 0
    for j in range(test_size):
            
        x = (test_X[j].flatten() - train_mean) / 255
        
        V = w_hidden @ x + bias_hidden
        y = tanh_act_fn(V)
        V_d = w_output @ y + bias_output
        y_d = sigmoid_act_fn(V_d)    
        
        actual_one_hot = np.empty((10, 1))
        desired_one_hot = np.empty((10, 1))
        actual_one_hot.fill(0.1)
        desired_one_hot.fill(0.1)
        actual_one_hot[np.argmax(y_d), :] = 0.9
        desired_one_hot[test_y[j], :] = 0.9
        
        e = (desired_one_hot - y_d.reshape((10,1))).reshape(10)
        
        test_energy += np.linalg.norm(e)**2
        if not np.array_equal(desired_one_hot, actual_one_hot):
            test_error += 1
                
    test_energy /= test_size
    
    test_energy_list.append(test_energy)
    test_error_list.append(test_error)
    print(f"Test Error: {test_error/test_size:.4f}\n")
    
    if train_energy_list[epoch] > train_energy_list[epoch-1]:
        eta *= 0.9
    if train_error_list[epoch]/train_size < 0.01:
        break   
    if epoch == 85:
        break
    epoch += 1
   
   
plt.plot(range(len(train_error_list)), train_error_list, 'b', label='Train Data')
plt.plot(range(len(test_error_list)), test_error_list, 'r', label='Test Data')
plt.axhline(y=500, c='black', linestyle=':')
plt.title('Error vs Epoch')
plt.xlabel('Number of Epochs')
plt.ylabel('Number of Misclassifications')
plt.legend()
plt.show() 

plt.plot(range(len(train_energy_list)), train_energy_list, 'b', label='Train Data')
plt.plot(range(len(test_energy_list)), test_energy_list, 'r', label='Test Data')
plt.title('Energy vs Epoch')
plt.xlabel('Number of Epochs')
plt.ylabel('Energy')
plt.legend()
plt.show()   
   
    

        

