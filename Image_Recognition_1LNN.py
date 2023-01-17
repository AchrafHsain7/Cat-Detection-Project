#getting started by importing the packages 
import numpy as np
import h5py
import copy
from PIL import Image
import matplotlib.pyplot as plt
from cats_functions import cats



#sigmoid for binary classification
def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s


#initializing the variables of my neural network 
def initialize(n_x, n_h, n_y):

    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.zeros((n_y,n_h)) * 0.01
    b2 = np.zeros((n_y,1))

    parameters = {"W1":W1,
                  "b1": b1,
                  "W2": W2,
                  "b2":b2}

    return parameters

def forward_propagation(X, Y,parameters):

    m = X.shape[1]
    #extracting parameters
    W1 = copy.deepcopy(parameters["W1"])
    b1 = parameters["b1"]
    W2 = copy.deepcopy(parameters["W2"])
    b2 = parameters["b2"]

    #forward!!
    Z1 = np.dot(W1, X) + b1
    #zeros = np.zeros(Z1.shape)
    A1 = np.tanh(Z1)  #tanh is used here
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

   
    cach = {"A1" : A1 ,
                "A2" : A2}

    return cach

def backward_propagation(X, Y,parameters ,cach):
    m = X.shape[1]

    #extracting the cach
    A1 = cach["A1"]
    A2 = cach["A2"]
    #extracting parameters
    W2 = parameters["W2"]

    #backward
    dZ2 = A2 - Y
    dW2 = 1/m * np.dot(dZ2, A1.T)
    db2 = 1/m * np.sum(dZ2, axis= 1, keepdims=True)
    dZ1 = np.multiply(np.dot(W2.T, dZ2) , 1-np.power(A1,2))
    dW1 = 1/m * np.dot(dZ1, X.T)
    db1 = 1/m * np.sum(dZ1, axis = 1, keepdims=True)

    grads = {
             "dW2" : dW2,
             "db2" : db2, 
             "dW1" : dW1,
             "db1" : db1
             }
    return grads

def gradient_descent(X, Y, parameters, cach, learning_rate ):
    #extracting
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    #backward propagation
    grads = backward_propagation(X,Y,parameters,cach)

    #Extracting
    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]

    #updating 
    W1 = W1 - learning_rate*dW1
    b1 = b1 - learning_rate*db1
    W2 = W2 - learning_rate*dW2
    b2 = b2 - learning_rate*db2

    parameters = {"W1":W1,
                  "b1": b1,
                  "W2": W2,
                  "b2":b2}

    return parameters

def compute_cost(Y, cach):

    #extract
    m = Y.shape[0]
    A2 = cach["A2"]

     #cost
    logprobs = np.multiply(np.log(A2),Y) + np.multiply(np.log(1-A2), 1-Y)
    cost = - 1/m * np.sum(logprobs)

    cost = float(np.squeeze(cost)) 

    return cost


def nn_model(X,Y,number_iterations, learning_rate, num_hidden_layers = 5, print_cost = False):
   

    #finding values of layers
    m = X.shape[1]
    n_x = X.shape[0]
    n_y = 1

    parameters = initialize(n_x, num_hidden_layers, n_y)
    costs = []

    for iter in range(number_iterations):
        cach = forward_propagation(X, Y, parameters)
        cost = compute_cost(Y, cach)
        costs.append(cost)
        parameters = gradient_descent(X, Y, parameters, cach, learning_rate)
        if(print_cost and (iter % 1000 == 0)):
            print("The cost at " + str(iter) + " iteration is: " + str(cost))

    d = {"costs": costs,
         "learning_rate" : learning_rate,
         "num_iterations": number_iterations}

    return parameters,d

def predict(X, parameters):

    m = X.shape[1]
    predictions = np.zeros((1,m))

    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    Z1 = np.dot(W1, X) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    for i in range(A2.shape[1]):

        if A2[0,i] > 0.5:
            predictions[0,i] = 1
        else:
            predictions[0,i] = 0

    return predictions

    
def predict_image(filename, parameters,classes, num_px = 64):
    # change this to the name of your image file
    my_image = "cats/" + filename   

    # We preprocess the image to fit your algorithm.
    fname =  my_image
    image = np.array(Image.open(fname).resize((num_px, num_px)))
    plt.imshow(image)
    image = image / 255.
    image = image.reshape((1, num_px * num_px * 3)).T
    my_predicted_image = predict(image, parameters)

    print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")
    plt.show()

    



def myNeuralNetwork():
    choice = 0

    X_train_orig, Y_train, X_test_orig, Y_test, classes = cats.load_h5_dataset()
    #preprocessing the data
    #  - flattening
    X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0], -1).T
    X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0], -1).T
    #unitizing the data
    X_train = X_train_flatten / 255
    X_test = X_test_flatten / 255

    


    while True:
        choice = input("Enter 1 to train the model\nEnter 2 to test model accuracy\nEnter 3 to predict on an image\n-->")

        try:
            if(choice == '1'):
                iter = int (input("Enter the number of iterations: "))
                learning_rate = float(input("Enter the learning rate: "))
                neurons = int(input("Enter the number of Neurons in the hidden layer: "))

                parameters,data = nn_model(X_train, Y_train, iter, learning_rate,num_hidden_layers= neurons,print_cost= True)
                cats.save_results2("parameters1LNN", parameters)
                print("Neural Network for recognizing cats built succesfully :) ")
            elif(choice == '2'):
                #accuracy test
                parameters = cats.load_results2("parameters1LNN")
                predictions_training = predict(X_train, parameters)
                training_accuracy = 100 - np.mean(np.abs( predictions_training - Y_train) * 100)

                predictions = predict(X_test, parameters)
                testing_accuracy = 100 - np.mean(np.abs( predictions - Y_test) * 100)

                print(str(training_accuracy) + "%")
                print(str(testing_accuracy) + "%")

                cats.plot_cost_graph(data)
            elif(choice == '3'):
                parameters = cats.load_results2("parameters1LNN")
                image = input("Enter the cat image name: ")
                image = image + ".jpg"
                predict_image(image, parameters, classes)

            else:
                print("lol....")
                return
        except:
            print("Oops something went wrong :( ...\n")


myNeuralNetwork()