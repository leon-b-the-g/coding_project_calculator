#First neural network

#Purpose: 


#Wrapping inputs for neural network
##Defining three vectors

#Then make dot products for the three vectors

#1. Multiply the first index of input_vector by the first index of weights_1.
#2. Multiply the second index of input_vector by the second index of weights_2.
#3. Sum the results of both multiplications.

#Importing modules
import numpy as np
import matplotlib.pyplot as plt
#import ipython

import os

#Providing inital test variables for the dot point computations
input_vector = np.array([1.66, 1.56])
weights_1 = np.array([1.45, -0.66])
bias = np.array([0.0])

#Calculating dot product (math written out for reference)
#first_indexes_mult = input_vector[0] * weights_1[0]
#second_indexes_mult = input_vector[1] * weights_1[1]
#dot_product_1 = first_indexes_mult + second_indexes_mult


#Calculating dot products with np

dot_product_1 = np.dot(input_vector, weights_1)

#dot_product_2 = np.dot(input_vector, weights_2)

#Check
print(f"Dot product of weight 1 and weight 2: {dot_product_1}")
#print(f"The dot product is: {dot_product_2}")


#Bernoulli distribution for estimating the probabilities of the relationship between the dot product sums 

#Sigmoid activation function for finalizing the output of the neural network (final layer)
#1/1+e^-x where e is the eulers number

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def predict(input_vector, weights, bias):
    #Layer 1 of model takes input vector, weights, and bias and computes the dot product
    dotproduct = np.dot(input_vector, weights) + bias
    #Layer 2 of model takes the output of layer 1 and applies the sigmoid function
    layer_1 = sigmoid(dotproduct)
    #returns the output of the first layer of linear function assessment
    return layer_1

#running a correct prediction, result should be above 0.5 to indicate the result is similar
layer_1 = predict(input_vector, weights_1, bias)

print(f"The prediction result is: {layer_1}")

#running a false prediction, result should be lower than 0.5 to indicate that the numbers in the input array arent similar

input_vector = np.array([2,1.5])

false_prediction = predict(input_vector, weights_1, bias)

print(f"The false prediction result is: {false_prediction}")



#Training the neural network to adjust the weights of the model to make better predictions 
#Adding a linear function assessment layer with a sigmoid function to the model
#Using the sigmoid function to determine the likelihood of the output of the model
#Using gradient decent (derivatives) and backpropagation algorithms to adjust the weights of the model to improve the prediction likelihood


#Gradient decent and backpropagation algorithms
# 1. compute a erroneous prediction, using mean square error function (find difference between prediction and target, multiply by itself)
# 2. reduce weights accordingly to adjust for weight of error (higher msqdif(error) means a more impactful error)
# 3. inform in which direction to adjust weight by calculating the derivative of the error calculation 


target = 0 
#Calculating the mean square difference
msqdif = np.square(false_prediction-target)
print(f"prediction: {false_prediction}; Error: {msqdif}")

#Calculating the error derivative
deriv = 2 * (false_prediction-target)

print(f"The derivative is {deriv}")

#Adjusting the weights of the model
weights_1 = weights_1 - deriv
layer_2 = predict(input_vector, weights_1,bias)
error = (layer_2 - target) ** 2 

print(f"prediction: {layer_2}; Error: {error}")


#Improve with alpha parameter
# determines learning rate; how much the weights are adjusted by the error derivative
#WHats the best learning rate? default values are 0.1, 0.01, and 0.001

#Using Stochastic gradient decent to prevent the model becoming overfitted
# Overfitted models are when a model isnt noticing new patterns in data, but rather memorizing the data it has seen and using that memory to provide predictions

#Due to chain rule in calculus (because we have a function composition: we cannot grab the derivative of a function without the derivative of the function it is composed of)
# , we need to calculate the derivative of the sigmoid function

#Need to multiply the derrivative of each function composition layer to get the derivative of the entire function composition
#Building th backward pass of the neural network
#Need derivative function of sigmoid

def sigmoid_deriv(x):
    return sigmoid(x) * (1 - sigmoid(x))


#Calculating the derivative of the error with respect to the weights
derror_dprediction = 2 * (false_prediction - target)

layer_1 = np.dot(input_vector, weights_1) + bias
#Calculating the derivative of the prediction with respect to the weights
dprediction_dlayer1 = sigmoid_deriv(layer_1)

#Setting the bias to 1 so that the transformations are linear and reduce error (makes calculation simpler too)
dlayer1_dbias = 1

#Calculating the derivative of the error with respect to the bias, complied by the chain rule
derror_dbias = (
    derror_dprediction * dprediction_dlayer1 * dlayer1_dbias
)

print("Derivative of Bias is:",derror_dbias)


#Alls works, writing class to wrap the functions

#Class for generating a neural network

class NeuralNetwork:
    def __init__(self, learning_rate):
        #Randomly generate some vectors for the weights
        self.weights = np.array([np.random.randn(), np.random.randn()])
        #Randomly generated bias, could provide user control here
        self.bias = np.random.randn()
        self.learning_rate = learning_rate

    #function for applying sigmoid function to the dot product of the input vector and the weights
    def sigmoid(self, x):
        return 1/(1+np.exp(-x))
    
    #function for applying the sigmoid function to the derivative of the sigmoid function
    def sigmoid_deriv(self, x):
        return sigmoid(x) * (1 - sigmoid(x))
    
    #function for running prediction on the model
    def predict(self,input_vector):
        #layer 1 of model takes input vector, weights, and bias and computes the dot product
        layer_1 = np.dot(input_vector,self.weights) + self.bias
        #layer 2 is a sigmoid function applied to the output of layer 1 (function composition)
        layer_2 = self.sigmoid(layer_1)
        prediction = layer_2
        return prediction
    
    #Function for backpassing the model to find:
        #derror_bias: derivative of the error with respect to the bias
        #derror_weights: derivative of the error with respect to the weights

    def _compute_gradients(self, input_vector, target):
        #making first layer of model (dot product)
        layer_1 = np.dot(input_vector, self.weights) + self.bias
        #making second layer of model (sigmoid function)
        layer_2 = self.sigmoid(layer_1)
        #Essentially the "predict" function, could scrap prediction function
        prediction = layer_2

        #Calculating the derivative of the error of the prediction with respect to the weights
        derror_dprediction = 2 * (prediction - target)

        #Calculating the derivative of layer 1 of prediction with respect to the weights
        dprediction_dlayer1 = self.sigmoid_deriv(layer_1)
        #Setting the bias to 1 so that the transformations are linear and reduce error (makes calculation simpler too)
        dlayer1_dbias = 1
        #Calculating the derivative of the error with respect to the bias, complied by the chain rule
        #I am not sure about this one:
        dlayer1_dweights = (0 * self.weights) + (1 * input_vector)

        #calculating the derivative of the error/bias relationship, chain rule style since we have all the derivatives now
        derror_dbias = (
            derror_dprediction * dprediction_dlayer1 * dlayer1_dbias
        )

        #Calculating the derivative of the error/weights relationship
        derror_dweights = (
            derror_dprediction * dprediction_dlayer1 * dlayer1_dweights
        )

        #Returning the derivatives
        return derror_dbias, derror_dweights
    

    #Function for updating the parameters of the model
    def _update_parameters(self, derror_dbias, derror_dweights):
        #Updating the bias and weights of the model, multiplying by the learning rate to adjust for error
        self.bias = self.bias - (derror_dbias * self.learning_rate)
        self.weights = self.weights - (derror_dweights * self.learning_rate)
        
    #Function for training the neural network
    def train(self, input_vectors, targets, iterations):
        #Tracking errors 
        cumulative_errors = []
        #Looping through the iterations to train the model
        for current_iteration in range(iterations):
            # Pick a data instance at random;COULD DEFINE DIFFERENT DATA INSTANCE HERE
            random_data_index = np.random.randint(len(input_vectors))

            input_vector = input_vectors[random_data_index]

            target = targets[random_data_index]

            # Compute the gradients and update the weights, function returns derror_dbias and derror_dweights
            derror_dbias, derror_dweights = self._compute_gradients(
                input_vector, target
            )
            # Update the model, adjusting for the assessments made in the backpass
            self._update_parameters(derror_dbias, derror_dweights)

            # Measure the cumulative error for all the instances
            if current_iteration % 100 == 0:
                cumulative_error = 0
                # Loop through all the instances to measure the error
                for data_instance_index in range(len(input_vectors)):
                    # Work through each instance and make input vector and target variables
                    data_point = input_vectors[data_instance_index]
                    target = targets[data_instance_index]

                    #generate a prediction for the data point and find error
                    prediction = self.predict(data_point)
                    error = np.square(prediction - target)

                    #Calculating the cumulative error
                    cumulative_error = cumulative_error + error
                cumulative_errors.append(cumulative_error)

        return cumulative_errors
#Class for training the neural network



#Creating a neural network instance, using a small sample data set to run the model

input_vectors = np.array(
    [
        [3, 1.5],
        [2, 1],
        [4, 1.5],
        [3, 4],
        [3.5, 0.5],
        [2, 0.5],
        [5.5, 1],
        [1, 1],
    ])

targets = np.array([0, 1, 0, 1, 0, 1, 1, 0])

learning_rate = 0.1

#Creating the neural network instance
neural_network = NeuralNetwork(learning_rate)

#Training the neural network
training_error = neural_network.train(input_vectors, targets, 10000)

#Plotting the training error

plt.plot(training_error)
plt.xlabel("Iterations")
plt.ylabel("Error across all training instances")
plt.savefig("cumulative_error.png")
    
        
            