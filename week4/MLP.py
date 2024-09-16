import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def forward_pass(x1, x2, weights_hidden, biases_hidden, weights_output, bias_output):
    # Compute hidden layer activations
    h1 = sigmoid(weights_hidden[0][0] * x1 + weights_hidden[0][1] * x2 + biases_hidden[0])
    h2 = sigmoid(weights_hidden[1][0] * x1 + weights_hidden[1][1] * x2 + biases_hidden[1])

    # Compute output layer activation
    y = sigmoid(weights_output[0] * h1 + weights_output[1] * h2 + bias_output)
    
    return h1, h2, y

def train(x1, x2, target, weights_hidden, biases_hidden, weights_output, bias_output, learning_rate):
    # Forward pass
    h1, h2, output = forward_pass(x1, x2, weights_hidden, biases_hidden, weights_output, bias_output)
    
    # Calculate error
    error = target - output
    
    # Backpropagation
    d_output = error * sigmoid_derivative(output)
    d_h1 = d_output * weights_output[0] * sigmoid_derivative(h1)
    d_h2 = d_output * weights_output[1] * sigmoid_derivative(h2)
    
    # Update weights and biases for output layer
    weights_output[0] += learning_rate * d_output * h1
    weights_output[1] += learning_rate * d_output * h2
    bias_output += learning_rate * d_output
    
    # Update weights and biases for hidden layer
    weights_hidden[0][0] += learning_rate * d_h1 * x1
    weights_hidden[0][1] += learning_rate * d_h1 * x2
    biases_hidden[0] += learning_rate * d_h1
    
    weights_hidden[1][0] += learning_rate * d_h2 * x1
    weights_hidden[1][1] += learning_rate * d_h2 * x2
    biases_hidden[1] += learning_rate * d_h2
    
    return weights_hidden, biases_hidden, weights_output, bias_output

def train_network(epochs, learning_rate):
    # Initialize weights and biases
    weights_hidden = [[0.5, -0.5], [-0.5, 0.5]]
    biases_hidden = [-0.5, -0.5]
    weights_output = [1, 1]
    bias_output = -0.5
    
    # Training data (x1, x2, target)
    training_data = [((1, 0), 1), ((0, 1), 0), ((1, 1), 0), ((0, 0), 0)]

    for _ in range(epochs):
        for (x1, x2), target in training_data:
            weights_hidden, biases_hidden, weights_output, bias_output = train(
                x1, x2, target, weights_hidden, biases_hidden, weights_output, bias_output, learning_rate
            )

    return weights_hidden, biases_hidden, weights_output, bias_output

def predict(x1, x2, weights_hidden, biases_hidden, weights_output, bias_output):
    _, _, y = forward_pass(x1, x2, weights_hidden, biases_hidden, weights_output, bias_output)
    return 1 if y > 0.5 else 0

# Train the network
epochs = 10000
learning_rate = 0.1
weights_hidden, biases_hidden, weights_output, bias_output = train_network(epochs, learning_rate)

# Example usage
print(predict(1, 0, weights_hidden, biases_hidden, weights_output, bias_output))  # Should print 1
print(predict(0, 1, weights_hidden, biases_hidden, weights_output, bias_output))  # Should print 0
