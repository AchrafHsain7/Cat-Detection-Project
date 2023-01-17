import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from cats_functions import cats




#version 1.3 of the NN model adding regularization, initialization optimization, and gradient checking
class NeuralNetworkModelCat_1_3:

    def __init__(self, layer_dims, learning_rate, num_iterations, lambd=0,regularization='L2'):
        self.layer_dims = layer_dims
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate
        self.lambd = lambd
        self.regularization = regularization

    #defining here the helper functions
    def reLU(self, Z):
        A = np.maximum(0, Z)
        
        return A
    
    def sigmoid(self, Z):
        A = 1 / (1 + np.exp(-Z))

        return A
    
    def deriv_reLU(self,dA, activation_cache):
        Z, _ = activation_cache
        dZ = np.array(dA, copy=True)
        dZ[ Z< 0] = 0

        return dZ
    
    def deriv_sigmoid(self, dA,activation_cache):
        Z,A = activation_cache
        dZ = dA * A*(1-A)

        return dZ

    def gradient_to_vector(self, gradients):
        #used to convert gradients from backprog toa vector to compare with estimation gradient
        count = 0
        L = len(self.layer_dims)
        for l in range(L-1):
            new_vectorW = np.reshape(gradients['dW'+str(l+1)], (-1,1))
            new_vectorb = np.reshape(gradients['db'+str(l+1)], (-1,1))
            new_vector = np.concatenate((new_vectorW, new_vectorb), axis=0)
            if count==0:
                theta = new_vector
            else:
                theta = np.concatenate((theta, new_vector), axis=0)
            count+=1
        return theta

    def param_to_vector(self, parameters):
        #used to convert parameters into a vector
        count = 0
        L = len(self.layer_dims)
        for l in range(L-1):
            new_vectorW = np.reshape(parameters['W'+str(l+1)], (-1,1))
            new_vectorb = np.reshape(parameters['b'+str(l+1)], (-1,1))
            new_vector = np.concatenate((new_vectorW, new_vectorb), axis = 0)
            if count==0:
                theta = new_vector
            else:
                theta = np.concatenate((theta, new_vector), axis=0)
            count+=1
        return theta
    
    def vector_to_param(self, theta, parameters):
        parameters_new = {}
        L = len(self.layer_dims)
        start = 0
        #print(len(theta))

        for l in range(L-1):
            #print(start)
            end = parameters['W'+str(l+1)].shape[0] * parameters['W'+str(l+1)].shape[1] + start
            #print(end)
            parameters_new['W'+str(l+1)] = theta[start:end].reshape(parameters['W'+str(l+1)].shape)
            start = end
            #print(start)
            end = parameters['b'+str(l+1)].shape[0] * parameters['b'+str(l+1)].shape[1] + start
            #print(end)
            parameters_new['b'+str(l+1)] = theta[start:end].reshape(parameters['b'+str(l+1)].shape)
            start = end
            #print('-----------------')

        return parameters_new

        


    def initialize_parameters(self):
        L = len(self.layer_dims)
        parameters = {}
        for l in range(L-1):
            parameters['W'+str(l+1)] = np.random.randn(self.layer_dims[l+1], self.layer_dims[l]) * np.sqrt(2/ self.layer_dims[l]) #He Initialization
            parameters['b' + str(l+1)] = np.zeros((self.layer_dims[l+1], 1))

        return parameters

    def normalize_data(X_train_o, X_test_o):

        X_train = X_train_o.reshape(X_train_o.shape[0], -1).T
        X_test = X_test_o.reshape(X_test_o.shape[0], -1).T

        X_train = np.divide(X_train, 255)
        X_test = np.divide(X_test, 255)

        return X_train, X_test

    def forward_propagation_1(self, W, b, A_prev, activation):
        #print(W.shape)
        #print(A_prev.shape)
        Z = np.dot(W, A_prev) + b
        if activation == 'reLU':
            A = self.reLU(Z)
        elif activation == 'sigmoid':
            A = self.sigmoid(Z)
        
        cache = (Z, A)

        return cache

    def forward_propagation_2(self, X_train, parameters):
        A_prev = X_train
        L = len(self.layer_dims)
        caches = {}

        for l in range(L-2):
            W = parameters['W'+str(l+1)]
            b = parameters['b'+str(l+1)]
            Z, A = self.forward_propagation_1(W, b, A_prev, 'reLU')

            linear_cache = (W, b, A_prev)
            activation_cahe = (Z, A)

            cache = (activation_cahe, linear_cache)
            caches['layer'+str(l+1)]=cache

            A_prev = A
        
        #the final layer
        W = parameters['W'+str(L-1)]
        b = parameters['b'+str(L-1)]
        Z, A = self.forward_propagation_1(W, b, A_prev, 'sigmoid')
        linear_cache = (W, b, A_prev)
        activation_cahe = (Z, A)
        cache = (activation_cahe, linear_cache)
        caches['layer'+str(L-1)] =cache

        return A, caches
    
    def compute_cost(self, A, Y_train, parameters, printit = False):
        m = Y_train.shape[0]
        L = len(self.layer_dims)
        reg = 0
        #adding L2 regularization
        log_cost = -1/m * np.sum(Y_train * np.log(A) + (1-Y_train)*np.log(1-A))
        if self.regularization == 'L2':
            for l in range(L-1):
                reg+= np.sum(np.square(parameters['W'+str(l+1)]))

            cost = log_cost + self.lambd/(2*m) * reg

        if printit:
            print('Cost before:' + str(log_cost))
            print('Cost after:' + str(cost))

        return cost

    def backward_propagation_1(self, dA, cache,m,activation):
        activation_cache, linear_cache = cache
        W, b, A_prev = linear_cache
        if activation == 'reLU':
            dZ = self.deriv_reLU(dA, activation_cache)
        elif activation == 'sigmoid':
            dZ = self.deriv_sigmoid(dA, activation_cache)

        dW = 1/m * np.dot(dZ, A_prev.T) + self.lambd/m * W
        db = 1/m * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)

        return dW, db, dA_prev

    def backward_propagation_2(self, Y_train, A, caches):

        grads = {}
        L = len(self.layer_dims)
        m = Y_train.shape[0]
        #reshape Y ?

        dA = -np.divide(Y_train, A) + np.divide((1-Y_train), (1-A))
        dW,db, dA_prev = self.backward_propagation_1(dA, caches['layer'+str(L-1)], m,'sigmoid')
        grads['dW'+str(L-1)] = dW
        grads['db'+str(L-1)] = db
        dA = dA_prev

        for l in reversed(range(L-2)):
            dW, db, dA_prev = self.backward_propagation_1(dA, caches['layer'+str(l+1)], m,'reLU')
            grads['dW'+str(l+1)] = dW
            grads['db'+str(l+1)] = db
            dA = dA_prev

        return grads

    def train_parameters(self, grads, parameters):
        L = len(self.layer_dims)

        for l in range(L-1):
            parameters['W'+str(l+1)] = parameters['W'+str(l+1)] - self.learning_rate*grads['dW'+str(l+1)]
            parameters['b'+str(l+1)] = parameters['b'+str(l+1)] - self.learning_rate*grads['db'+str(l+1)]

        return parameters  

    def train_model(self, X, Y, print_cost=False):  

        costs = []
        parameters = self.initialize_parameters()
        for i in range(self.num_iterations):
            Activation, caches = self.forward_propagation_2(X, parameters)
            cost = self.compute_cost(Activation, Y, parameters)
            costs.append(cost)
            grads = self.backward_propagation_2(Y, Activation, caches)
            parameters = self.train_parameters(grads, parameters)
            if(print_cost and i%1000==0):
                print('Cost at iteration '+str(i)+": " + str(cost))

            model = {"learning_rate" :  self.learning_rate,
                    "costs" : costs}
            
        return parameters, model

    def gradient_check(self, X, Y,parameters, grads, epsilon=1e-7):

        #initializing
        new_params = self.param_to_vector(parameters)
        param_num = new_params.shape[0]
        gradients = self.gradient_to_vector(grads)
        J_plus = np.zeros((param_num, 1))
        J_minus = np.zeros((param_num, 1))
        grads_approx = np.zeros((param_num, 1))
        print(param_num)

        #computing grad approx
        for i in range(param_num):
            theta_plus = np.copy(new_params)
            theta_plus[i] = theta_plus[i] + epsilon
            param = self.vector_to_param(theta_plus, parameters)
            A,_ = self.forward_propagation_2(X, param)
            J_plus[i] = self.compute_cost(A, Y, param)

            theta_minus = np.copy(new_params)
            theta_minus[i] = theta_minus[i] - epsilon
            param = self.vector_to_param(theta_minus, parameters)
            A,_ = self.forward_propagation_2(X, param)
            J_minus[i] = self.compute_cost(A, Y, param)

            grads_approx[i] = (J_plus[i] - J_minus[i])/(2*epsilon)
        
        #computing the difference
        numerator = np.linalg.norm(grads_approx - gradients)
        denominator = np.linalg.norm(gradients) + np.linalg.norm(grads_approx)
        error = numerator / denominator

        return error

    def accuracy(self, X, Y, parameters, printing=False):
        m = X.shape[1]
        n = len(parameters) // 2 # number of layers in the neural network
        p = np.zeros((1,m))
        
        # Forward propagation
        probas, caches = self.forward_propagation_2(X, parameters)

        
        # convert probas to 0/1 predictions
        for i in range(0, probas.shape[1]):
            if probas[0,i] > 0.5:
                p[0,i] = 1
            else:
                p[0,i] = 0
        
        if printing:
            print("Accuracy: "  + str(np.sum((p == Y)/m)))
            
        return p
    
    def predict(self, X, parameters):

        m = X.shape[1]
        predictions = np.zeros((1,m))

        AL, caches = self.forward_propagation_2(X, parameters)
        for i in range(AL.shape[1]):

            if AL[0,i] > 0.5:
                predictions[0,i] = 1
            else:
                predictions[0,i] = 0

        return predictions


    #predicting an image
    def imagePrediction(self, my_image, parameters, classes, num_px = 64):
        fname =  'cats/' + my_image
        image = np.array(Image.open(fname).resize((num_px, num_px)))
        plt.imshow(image)
        image = image / 255.
        image = image.reshape((1, num_px * num_px * 3)).T
        my_predicted_image = self.predict(image, parameters)

        print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")
        plt.show()
    
    def test_images(self, parameters, classes, num_px = 64):
      images = ['cats/cat1.jpg', 'cats/cat2.jpg', 'cats/cat3.jpg', 'cats/cat4.jpg', 'cats/cat5.jpg', 'cats/Shrekk.jpg']
      images_res = [1,1,1,1,1,0]
      accuracy = 0
      for i in range(len(images)):
        fname =  images[i]
        image = np.array(Image.open(fname).resize((num_px, num_px)))
        plt.imshow(image)
        image = image / 255.
        image = image.reshape((1, num_px * num_px * 3)).T
        my_predicted_image = self.predict(image, parameters)
        if np.squeeze(my_predicted_image) == images_res[i]:
          accuracy+=1

        print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")
        plt.show()
      
      return accuracy/6 * 100

            




     



def test():
    X_train, Y_train, X_test, Y_test, classes = cats.load_h5_dataset()
    X_train, X_test = NeuralNetworkModelCat_1_3.normalize_data(X_train, X_test)
    layers = [X_train.shape[0], 20, 4, 1]
    model = NeuralNetworkModelCat_1_3(layers, 7000, 0.0075, 0.7)
    """""
    param = model.initialize_parameters()
    print('Test2 success')
    A, caches = model.forward_propagation_2(X_train, param)
    print('Test3 success')
    cost = model.compute_cost(A, Y_train, param, False)
    print('Test4 success')
    grads = model.backward_propagation_2(Y_train, A, caches)
    print('Test5 success')
    param = model.train_parameters(grads, param)
    print('Test6 success')
    margin = model.gradient_check(X_train,Y_train, param, grads)
    print(margin)
    print('Test7 success')"""
    param = model.train_model(X_train, Y_train, True)
    print("success")
    model.accuracy(X_test,Y_test, param, True)

    
    


#test()



















#classic non regularized or optimized model for image detection
class NeuralNetworkModelCat_1_0:

    def __init__(self, layerDimensions, learningRate, numberIterations):

        self.layerDimensions = layerDimensions
        self.learningRate = learningRate
        self.numberIterations = numberIterations

        

    def sigmoid(self,Z):
        A = 1 / (1+ np.exp(-Z))
        cache = Z

        return A, cache

    def relu(self,Z):
        #if Z > 0 (Z>0)=1 else  = 0
        A = np.maximum(0, Z)

        #assert(A.shape == Z.shape)
        cache = Z

        return A, cache

    def backwardSigmoid(self, dA, activationCache):
        
        #calculating dZ because the derivative of sigmoid(z) = a(1-a)
        Z= activationCache
        s = 1 / (1 + np.exp(-Z))
        dZ = dA * s * (1 - s)
        #assert (dZ.shape == Z.shape)

        return dZ

    def backwardRelu(self, dA, activationCache):
        Z= activationCache
        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0
        #calculating dZ and derivative of ReLU is 0 Z<0 else dZ = 
        #assert (dZ.shape == Z.shape)

        return dZ

    def preprocessData(X_train , X_test):
        #flattening the image matrix into a vector containing the features X anf m the number of examples
        X_train_flat = X_train.reshape(X_train.shape[0], -1).T
        X_test_flat = X_test.reshape(X_test.shape[0], -1).T

        #Uniforming the data of pixels by dividing everything by 255
        X_train = X_train_flat / 255
        X_test = X_test_flat / 255

        return X_train, X_test



    def initializeParameters(self):
        #layer_dims is an array containing dimesions for all layers in the order
        #we will return a dictionary containing all parameters for all layers
        parameters = {}
        L = len(self.layerDimensions)

        #iterating over the different layer and initializing the parameters
        for l in range(1,L):
            parameters['W' + str(l)] = np.random.randn( self.layerDimensions[l],  self.layerDimensions[l-1]) / np.sqrt(self.layerDimensions[l-1])
            parameters['b' + str(l)] = np.zeros(( self.layerDimensions[l], 1))

        
        return parameters

    def linearForward(self,A_prev, W, b):
        #this function is general and wont change and just calculate Z while keeping track of the cach for backward propagation
        #A is the activations from the previous layer

        #calculating yhe linear result Z
        Z = np.dot(W, A_prev) + b

        linearCache = (A_prev, W, b)

        return Z, linearCache

    def activationForward(self, A_prev, W, b, activation):
        #this function compute the activations of the current layer while taking into consideration the activation function used

        #computing Z and getting the cach
        Z, linearCache =  self.linearForward(A_prev, W, b)
        #sigmoid case probably the output layer
        if(activation == "sigmoid"):
            #computing Z and getting the cach
            Z, linearCache =  self.linearForward(A_prev, W, b)
            A, activationCache =  self.sigmoid(Z)

        #relu implementation
        elif(activation == "relu"):
            #computing Z and getting the cach
            Z, linearCache =  self.linearForward(A_prev, W, b)
            A, activationCache =  self.relu(Z)
        
        #sending both caches for backward propagation
        cache = (linearCache, activationCache)
        
        return A, cache


    def forwardPropagation(self, X, parameters):
        #implemetation of forward propagation used for predicting and training the odel
        L = len(parameters) // 2
        A = X
        caches = []

        #calculating all the activations across the layers using a reLu activation function
        for l in range(1,L):
            #updating the activations for next layer
            A_prev = A
            A, cache =  self.activationForward(A_prev, parameters["W" + str(l)], parameters["b"+str(l)], "relu")
            caches.append(cache)


        #the output layer is calculated alone beacsue the only one that need a sigmoid activation function and the one that we return
        AL, cache =  self.activationForward(A, parameters["W" + str(L)], parameters["b"+str(L)], "sigmoid")
        caches.append(cache)

        return AL, caches

    def costCalculation(self, AL, Y):
        #number of training examples m
        m = Y.shape[1]
        cost = -1/m * np.sum(np.multiply(Y,np.log(AL)) + np.multiply(1-Y,np.log(1-AL)) ) 

        cost = np.squeeze(cost)

        return cost

    def linearBackward(self, dZ, linearCache):
        #doing a backward step for one layer
        #input getting dZ and cahe(A_prev, W, b)
        A_prev, W, b = linearCache
        m = A_prev.shape[1]

        dW = 1/m * np.dot(dZ, A_prev.T)
        db = 1/m * np.sum(dZ, axis =1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)

        return dA_prev, dW, db

    def linearActivationBackward(self, dA,cache ,activation):
            
        linearCache, activationCache = cache

            #calculating all dZ based on the activation function used
        if(activation == "relu"):
            dZ =  self.backwardRelu(dA, activationCache)
            #calculating the gradients based on the previous function
            dA_prev, dW, db =  self.linearBackward(dZ, linearCache)

        
        elif(activation == "sigmoid"):
            dZ =  self.backwardSigmoid(dA, activationCache)
            #calculating the gradients based on the previous function
            dA_prev, dW, db =  self.linearBackward(dZ, linearCache)
        
        return dA_prev, dW, db 



    def backwardPropagation(self, AL, Y, caches):
        #getting the number of training examples
        m = AL.shape[1]
        #the number of layers
        L = len(caches)
        grads = {}
        Y = Y.reshape(AL.shape)

        #the derivative of AL in respect to the cost function
        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

        #backward for the L-1 layer because use of sigmoid
        current_cache = caches[L-1]
        dA_prev_tmp, dW_tmp, db_tmp =  self.linearActivationBackward(dAL, current_cache, "sigmoid")
        grads["dW"+str(L)] = dW_tmp
        grads["db"+str(L)] = db_tmp
        grads["dA"+str(L-1)] = dA_prev_tmp

        #backward for the rest of the layers
        for l in reversed(range(L-1)):
            current_cache = caches[l]
            dA_prev_tmp, dW_tmp, db_tmp =  self.linearActivationBackward(grads["dA"+str(l+1)], current_cache, "relu")
            grads["dW"+str(l+1)] = dW_tmp
            grads["db"+str(l+1)] = db_tmp
            grads["dA"+str(l)] = dA_prev_tmp

        return grads

    def updateParameters(self, parameters, grads):
        params = parameters.copy()
        L = len(params) // 2

        #updating and training everything
        for l in range(L):
            params["W"+str(l+1)] = params["W"+str(l+1)] - self.learningRate * grads["dW"+str(l+1)]
            params["b"+str(l+1)] = params["b"+str(l+1)] - self.learningRate * grads["db"+str(l+1)]

        return params

    def trainModel(self, X, Y, printing = False):

        parameters = self.initializeParameters()
        costs = []
        Y = Y.reshape(1,Y.shape[0])

        for i in range( self.numberIterations):
            #forward
            AL, caches =  self.forwardPropagation(X, parameters)

            #cost
            cost =  self.costCalculation(AL,Y)
            costs.append(cost)

            if(printing and i%500==0 or i ==  self.numberIterations-1):
                print("Cost at iterations " + str(i) + ": " + str(cost))
                

            #backward
            grads =  self.backwardPropagation(AL, Y, caches)

            #update
            parameters =  self.updateParameters(parameters, grads)

        model = {"learning_rate" :  self.learningRate,
                    "costs" : costs}
        
        return parameters, model

  
    #testing the accuracy compared to the testing set
    def accuracy(self, X, Y, parameters, printing=False):
        m = X.shape[1]
        n = len(parameters) // 2 # number of layers in the neural network
        p = np.zeros((1,m))
        
        # Forward propagation
        probas, caches = self.forwardPropagation(X, parameters)

        
        # convert probas to 0/1 predictions
        for i in range(0, probas.shape[1]):
            if probas[0,i] > 0.5:
                p[0,i] = 1
            else:
                p[0,i] = 0
        
        if printing:
            print("Accuracy: "  + str(np.sum((p == Y)/m)))
            
        return p

    #predicting the result of an X
    def predict(self, X, parameters):

        m = X.shape[1]
        predictions = np.zeros((1,m))

        AL, caches = self.forwardPropagation(X, parameters)
        for i in range(AL.shape[1]):

            if AL[0,i] > 0.5:
                predictions[0,i] = 1
            else:
                predictions[0,i] = 0

        return predictions


    #predicting an image
    def imagePrediction(self, my_image, parameters, classes, num_px = 64):
        fname =  'cats/' + my_image
        image = np.array(Image.open(fname).resize((num_px, num_px)))
        plt.imshow(image)
        image = image / 255.
        image = image.reshape((1, num_px * num_px * 3)).T
        my_predicted_image = self.predict(image, parameters)

        print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")
        plt.show()
    
    def test_images(self, parameters, classes, num_px = 64):
      images = ['cats/cat1.jpg', 'cats/cat2.jpg', 'cats/cat3.jpg', 'cats/cat4.jpg', 'cats/cat5.jpg', 'cats/Shrekk.jpg']
      images_res = [1,1,1,1,1,0]
      accuracy = 0
      for i in range(len(images)):
        fname =  images[i]
        image = np.array(Image.open(fname).resize((num_px, num_px)))
        plt.imshow(image)
        image = image / 255.
        image = image.reshape((1, num_px * num_px * 3)).T
        my_predicted_image = self.predict(image, parameters)
        if np.squeeze(my_predicted_image) == images_res[i]:
          accuracy+=1

        print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")
        plt.show()
      
      return accuracy/6 * 100

    def saveParameters(self, filename, parameters):
        L = len(parameters)//2
        W = []
        b = []
        for i in range(L):
            W.append(parameters["W"+str(i+1)])
            b.append(parameters["b"+str(i+1)])

        np.savez(filename, W = W, b = b)

    def loadParameters(self, filename):
        fileN = filename + ".npz"
        data = np.load(fileN)
        W = data["W"]
        b = data["b"]
        parameters = {}
        L = len(W)
        for i in range(L):
            parameters["W"+str(i+1)] = W[i]
            parameters["b"+str(i+1)] = b[i]

        return parameters
