import math

# Helper functions for the neural network
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def normalize(value, min_value, max_value):
    return (value - min_value) / (max_value - min_value)

def denormalize(normalized_value, min_value, max_value):
    return normalized_value * (max_value - min_value) + min_value

# Neural network functions
def forward_pass(x, weights_hidden, biases_hidden, weights_output, bias_output):
    # Compute hidden layer activations
    h = [sigmoid(sum(w * xi for w, xi in zip(weights_hidden[i], x)) + biases_hidden[i])
         for i in range(len(biases_hidden))]

    # Compute output layer activation
    y = sigmoid(sum(w * hi for w, hi in zip(weights_output, h)) + bias_output)
    
    return h, y

def train(x, target, weights_hidden, biases_hidden, weights_output, bias_output, learning_rate):
    # Forward pass
    h, output = forward_pass(x, weights_hidden, biases_hidden, weights_output, bias_output)
    
    # Calculate error
    error = target - output
    
    # Backpropagation
    d_output = error * sigmoid_derivative(output)
    d_h = [d_output * weights_output[i] * sigmoid_derivative(h[i]) for i in range(len(h))]
    
    # Update weights and biases for output layer
    for i in range(len(weights_output)):
        weights_output[i] += learning_rate * d_output * h[i]
    bias_output += learning_rate * d_output
    
    # Update weights and biases for hidden layer
    for i in range(len(weights_hidden)):
        for j in range(len(weights_hidden[i])):
            weights_hidden[i][j] += learning_rate * d_h[i] * x[j]
        biases_hidden[i] += learning_rate * d_h[i]
    
    return weights_hidden, biases_hidden, weights_output, bias_output

def train_network(X, y, epochs, learning_rate):
    # Initialize weights and biases
    input_size = len(X[0])
    hidden_layer_size = 5  # Number of neurons in hidden layer (can be tuned)
    output_size = 1  # Single output for regression
    
    weights_hidden = [[0.5 for _ in range(input_size)] for _ in range(hidden_layer_size)]
    biases_hidden = [0.5 for _ in range(hidden_layer_size)]
    weights_output = [0.5 for _ in range(hidden_layer_size)]
    bias_output = 0.5
    
    for _ in range(epochs):
        for xi, target in zip(X, y):
            weights_hidden, biases_hidden, weights_output, bias_output = train(
                xi, target, weights_hidden, biases_hidden, weights_output, bias_output, learning_rate
            )
    
    return weights_hidden, biases_hidden, weights_output, bias_output

def predict(x, weights_hidden, biases_hidden, weights_output, bias_output):
    _, y = forward_pass(x, weights_hidden, biases_hidden, weights_output, bias_output)
    return y

# Data preprocessing
data = [
    [3, 1500, 10, 'Good', 40000],
    [3, 2000, 5, 'Medium', 32000],
    [4, 2500, 10, 'Poor', 25000],
    [2, 1000, 3, 'Good', 25000],
    [2, 1200, 5, 'Medium', 22000],
    [4, 2200, 15, 'Good', 50000],
    [2, 1000, 10, 'Poor', 12000],
    [5, 3000, 15, 'Good', 60000],
    [3, 1600, 12, 'Good', 35000],
    [4, 2600, 8, 'Medium', 50000],
    [3, 1750, 6, 'Good', 38000],
    [2, 900, 10, 'Good', 20000],
    [2, 1100, 12, 'Poor', 15000],
    [3, 1800, 18, 'Medium', 27000]
]

# Extract features and target
X_raw = [row[:4] for row in data]
y = [row[4] for row in data]

# Encode categorical feature 'Location'
location_mapping = {'Good': 0, 'Medium': 1, 'Poor': 2}
X = [[x1, x2, x3, location_mapping[x4]] for x1, x2, x3, x4 in X_raw]

# Normalize numerical features
carpet_area_max = max(x[1] for x in X)
carpet_area_min = min(x[1] for x in X)
years_old_max = max(x[2] for x in X)
years_old_min = min(x[2] for x in X)

X = [
    [
        x[0],  # Number of bedrooms
        normalize(x[1], carpet_area_min, carpet_area_max),  # Carpet Area
        normalize(x[2], years_old_min, years_old_max),  # Years Old
        x[3]  # Location (already encoded)
    ]
    for x in X
]

# Normalize target variable (rent)
rent_max = max(y)
rent_min = min(y)
y = [normalize(rent, rent_min, rent_max) for rent in y]

# Train the network
epochs = 10000
learning_rate = 0.01
weights_hidden, biases_hidden, weights_output, bias_output = train_network(X, y, epochs, learning_rate)

# Make predictions
predicted = [denormalize(predict(x, weights_hidden, biases_hidden, weights_output, bias_output), rent_min, rent_max) for x in X]

# Print results
for i, (original, pred) in enumerate(zip(y, predicted)):
    print(f"House {i+1}: True Rent = {denormalize(original, rent_min, rent_max)}, Predicted Rent = {pred:.2f}")
