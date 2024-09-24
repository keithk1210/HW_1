import numpy as np
import matplotlib.pyplot as plt

# Data generation
np.random.seed(0)
num_observations = 500

x1 = np.random.multivariate_normal([0, 0], [[1, .75], [.75, 1]], num_observations)
x2 = np.random.multivariate_normal([1, 4], [[1, .75], [.75, 1]], num_observations)

features = np.vstack((x1, x2)).astype(np.float32)
labels = np.hstack((np.zeros(num_observations), np.ones(num_observations)))

# Sigmoid function (5 pts)
def sigmoid(scores):
    # --------------------
    # Write your code here:
    probs = 1 / (1 + np.exp(-scores))
    # --------------------
    return probs

# Build the LR model (10 pts for CSE 426 while 15 pts for CSE 326)
def logistic_regression(features, target, num_steps, learning_rate, add_intercept=False):
    if add_intercept:
        intercept = np.ones((features.shape[0], 1))
        features = np.hstack((intercept, features))

    weights = np.zeros(features.shape[1])

    losses = []
    # Update weights with gradient
    for step in range(num_steps):
        # --------------------
        # Write your code here:

        # Apply the weights to the inputs

        weighted_input = np.dot(features, weights)

        # Apply the sigmoid function on the weighted inputs

        predictions = sigmoid(weighted_input)

        # Compute the difference between the predictions and the target, (the error)

        error = predictions - target

        # Use the error to calculate the gradient

        gradient = (np.dot(features.T, error)) / len(target)
        
        # --------------------
        weights -= learning_rate * gradient
        
        losses.append(get_loss(weights, features, target))
        
    return weights, losses

# Implement the logistic model with the weights
def predict(features, weights, add_intercept=False):
    if add_intercept:
        intercept = np.ones((features.shape[0], 1))
        features = np.hstack((intercept, features))
    scores = np.dot(features, weights)
    predictions = sigmoid(scores)
    return predictions > 0.5

# Function to compute the loss
def get_loss(weights, features, target):
    scores = np.dot(features, weights)
    predictions = sigmoid(scores)
    loss = -np.mean(target * np.log(predictions) + (1 - target) * np.log(1 - predictions))
    return loss

# Train the LR and print the updated model weights (5 pt)
weights, losses = logistic_regression(features, labels, num_steps=30, learning_rate=5e-5, add_intercept=True)
print(f'Weights: {weights}')


# Make predictions
predictions = predict(features, weights, add_intercept=True)
accuracy = (predictions == labels).mean()
print(f"Training Accuracy: {accuracy * 100:.2f}%")

# Function to plot the loss over iterations
def plot_losses(losses):
    plt.plot(losses)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss over iterations')
    plt.show()
# Plot the loss
plot_losses(losses)