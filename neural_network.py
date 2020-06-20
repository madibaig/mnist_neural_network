"""A from-scratch simple feedforward neural network utilising the square error
cost function for the MNIST data set. An network instance can be made from the
NeuralNetwork class which can then be trained to be able to identify images
from the MNIST data set. """

import numpy as np
import pprint
import random

class NeuralNetwork():
    """Method of a neural network:
    1. Intialise number of layers, and neurons for each layer DONE
        - 784 input neurons
        - 10 output neurons
    2. Initialise the weights and biases randomly according to the neurons DONE
    3. Train the networks' weights and biases using gradient descent and bp
        - Divide all the training data into minibatches
        - Each batch is for an epoch which is passed into batch_gradient_descent
            - batch_gradient_descent calculates the partial differentials for all
            weights and biases in respect to the cost using backpropagation.
            - These partial differentials are calculated for each training example
            in the batch
            - All the weights and biases are adjusted by their average partial
            derivatives, completing the batch and epoch
        - (optional) Evaluate performance using test_data if specified
        - Do this for all epochs until finished"""

    def __init__(self, neurons_per_layer):
        """neurons_per_layer is a list [neurons in input layer, neurons in
        hidden layer, neurons in output layer]."""
        self.neurons_per_layer = neurons_per_layer
        self.layers = len(neurons_per_layer)

        #weights between input layer and 2[[[w0.0, w0.1, w0.2],[w1.0, w1.1, w1.2],[],[],[],[]...],
        #weights between layer 2 and 3 [[w0.0, w0.1],[],[],[]...]]
        self.weights = [np.random.randn(neurons_per_layer[L], neurons_per_layer[L-1])
                        for L in range(1, len(neurons_per_layer))]

        #Biases has a 1-dimensional np array with biases for each neuron in each
        #layer (apart from the input layer)
        self.biases = [np.random.randn(neurons, 1) for neurons in neurons_per_layer[1:]]

    def feedforward(self, a):
        """Given the input x to the first layer, this calculates and returns the
        output layer (in a list) using the weights and biases."""

        #Calculating the activations of each layer accumulating to the output
        for layer in range(0, self.layers-1):
            #sigmoid(w*a + b)
            a = self.sigmoid(np.dot(self.weights[layer], a) + self.biases[layer])

        return a


    def backpropagation(self, x, y):
        """Implements the backpropagation algorithm to calculate dcost_dweight and
        dcost_dbias and returns the matrices.
        Cost function is 1/2(y - a)^2
        Method:
        1. Feedforward x but manually saving each layer's activation and z
        2. Compute the error of the output layer
            - Derivative of cost with respect to output layer's a *
            derivative of sigmoid(output layer's z)
        3. Find the errors of the previous layers using output layer's error
        4. Assign the layer errors to corresponding bias derivatives
        5. Compute and assign the weight derivatives
        6. Return the weight and biases derivatives as ndarrays"""

        a = x
        a_by_layer = [x] #Holds activations of each layer for later equations
        z_by_layer = [] #Holds z of each layer for later equations

        #1. Feedforward layer by layer
        for layer in range(0, self.layers-1):
            z = np.dot(self.weights[layer], a) + self.biases[layer]
            #z_by_layer.append(z.reshape(self.neurons_per_layer[layer+1], 1))
            z_by_layer.append(z)
            a = self.sigmoid(z)
            #a_by_layer.append(a.reshape(self.neurons_per_layer[layer+1], 1))
            a_by_layer.append(a)

        #2. Compute error of output layer
        output_layer_error = self.cost_deriv(y, a_by_layer[-1]) * self.sigmoid_prime(z_by_layer[-1])

        #3. Find errors of previous layers
        error_by_layer = [output_layer_error]
        for l in range(1, self.layers - 1):
            error_by_layer.append(np.dot(self.weights[-l].transpose(), error_by_layer[l-1]) * self.sigmoid_prime(z_by_layer[-l-1]))
            
        #4. Assign the layer errors in correct order (reversed) to biases derivs
        dcost_dbiases = [error_by_layer[-(layer+1)] for layer in range(len(error_by_layer))]

        #5. Assigning weight derivatives
        dcost_dweights = [np.zeros(np.shape(w)) for w in self.weights]
        for layer in range(self.layers-1):
            for j in range(len(self.weights[layer])):
                for k in range(len(self.weights[layer][j])):
                    dcost_dweights[layer][j][k] = a_by_layer[layer][k] * dcost_dbiases[layer][j]

        #6. Return the partial derivatives in shapes of self.weights and biases
        return dcost_dweights, dcost_dbiases

    def batch_gradient_descent(self, batch, learn_rate):
        """Function that uses gradient descent to adjust the weights and biases
        according to the average partial derivatives over the whole batch.
        Method:
        1. Create ndarrays in shape of weights and biases to hold derivatives
        2. For each example in batch:
            - Pass x and y into backpropagation and assign the derivatives to new
            ndarrays
            - Add the derivatives to the previous ndarrays of the weights and
            biases, creating a total
        3. Average out the weights and biases differential totals
        4. Minus the averages for each weight and bias using the learn_rate"""

        total_weight_derivs = [np.zeros(np.shape(w)) for w in self.weights]
        total_bias_derivs = [np.zeros(np.shape(b)) for b in self.biases]

        for example in batch:
            #Backpropagate example
            weight_derivs, bias_derivs = self.backpropagation(example[0], example[1])
            #Add derivatives to totals
            total_weight_derivs = [twd + wd for twd, wd in
                                    zip(total_weight_derivs, weight_derivs)]
            total_bias_derivs = [tbd + bd for tbd, bd in
                                    zip(total_bias_derivs, bias_derivs)]

        #Length of batches to average the totals
        n = len(batch)

        #Gradient descent for all weights and biases
        self.weights = [w - learn_rate * (twd / n) for w, twd in
                        zip(self.weights, total_weight_derivs)]
        self.biases = [b - learn_rate * (tbd / n) for b, tbd in
                        zip(self.biases, total_bias_derivs)]

    def train(self, epochs, learn_rate, batch_size, train_data, test_data=None):
        """Function called to train the neural network which splits up the
        training data into batches and does batch_gradient_descent for each epoch,
        using backpropagation."""

        for epoch in range(epochs):
            random.shuffle(train_data)
            batches = [train_data[part:part+batch_size] for part in range(0, len(train_data), batch_size)]

            for batch in batches:
                self.batch_gradient_descent(batch, learn_rate)

            if(test_data):
                print(f"Epoch {epoch + 1} complete. Accuracy: {self.evaluate(test_data)}")
            else:
                print(f"Epoch {epoch + 1} complete. No evaluation specified.")

        print("Training complete.")

    def evaluate(self, test_data):
        """Function that returns percentage of correct guesses of the neural
        network based on the test_data given. Returns a double. For each point
        in test_data, it feedforwards it, compares it to y and adds it to the
        total if the guess was correct."""

        correct_guesses = 0

        for test in test_data:
            result = self.feedforward(test[0]) #Input x from test

            #Seeing if the guess is correct by comparing to y
            guess = 0
            #Finding the actual guess of the output layer
            for neuron in range(len(result)):
                if(result[neuron] > result[guess]):
                    guess = neuron
            #Comparing the guess to y
            if(guess == np.where(test[1] == 1)[0][0]):
                print(guess, np.where(test[1] == 1)[0][0])
                correct_guesses += 1

        return correct_guesses / len(test_data)

    def cost_deriv(self, y, a):
        """Returns the derivative of the cost with respect to a as a layer."""
        deriv = []
        for i in range(len(y)):
            deriv.append(a[i] - y[i])

        return np.array(deriv)

    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def sigmoid_prime(self, z):
        """Derivative of sigmoid."""
        return self.sigmoid(z) * (1.0 - self.sigmoid(z))

