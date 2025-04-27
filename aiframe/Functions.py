import numpy as np

def squared_loss(predicted, expected):
    return 2 * (predicted - expected)

def calculate_layer_backward_values(weights, pass_values, activation):
    return np.dot(weights.T * activation, pass_values)

def update_gradient(pass_values, layer_output):
    return pass_values * layer_output[:, None], pass_values