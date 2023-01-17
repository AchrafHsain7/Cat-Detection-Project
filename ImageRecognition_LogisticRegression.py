
#importing the packages
import numpy as np
import matplotlib.pyplot as plt
import h5py
import copy
from PIL import Image
from cats_functions import cats
#from scipy import ndimage



#function that plot a picture given the index
def show_picture(index, X, Y):
    plt.imshow(X[index])
    print("y= " + str(Y[index]))



#defining the activation_fct function for later
def activation_fct(X):
    result = 1 / (1 + np.exp(-X))
    
    return result

#function that initialize all the parameters to 0
def initialize_param(dimension):

    w = np.zeros((dimension, 1))
    b = 0.0

    return w,b

#forward and backward propagation to calculate the cost and the gradients
def propagate(w,b, X, Y):

    m = X.shape[1] #the size of the dataset/ number of images

    #forward propagation
    A = activation_fct(np.dot(w.T, X) + b) #calculating the activation value y hat
    cost =  1/m * np.sum(-(Y*np.log(A) + (1-Y)*np.log(1-A)))  #calculating the cost of the current state

    #backward propagation
    dw = 1/m * np.dot(X, (A - Y).T)
    db = 1/m * np.sum((A-Y))

    cost = np.squeeze(np.array(cost))
    grads = {
        "dw": dw ,
        "db": db 
    }

    return grads, cost


#function implementing the gradient descent algorithme to optimize w and b
def optimize(w, b, X, Y, num_iterations=2500, learning_rate = 0.005, print_cost=False):

    costs = []
    #deep copy for creating a datastructure similar to the original the populating it with copied images
    w = copy.deepcopy(w)
    b = copy.deepcopy(b) 

    for iter in range(num_iterations):
        grads, cost = propagate(w,b, X, Y)

        dw = grads["dw"]
        db = grads["db"]

        w = w - learning_rate * dw
        b = b - learning_rate * db

        #recording the cost every 100 iterations and optionally printing it
        if(iter % 100 == 0):
            costs.append(cost)

            if print_cost:
                print("the cost at iteration" + str(iter) + " is: " + str(cost))

    params = {"w": w, "b": b}
    grads = {"dw": dw, "db": db}

    return params, grads, costs

#function that predict giventhe values w and b if the image is a 1(is a "target") or 0(is not a "target")
def predict(w,b, X):

    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape((X.shape[0], 1))

    A = activation_fct(np.dot(w.T, X) + b)

    for i in range(A.shape[1]):

        if A[0,i] > 0.5:
            Y_prediction[0,i] = 1
        else:
            Y_prediction[0,i] = 0
    
    return Y_prediction

#the final model
def model(X_train, Y_train ,X_test, Y_test, num_iterations=2500, learning_rate=0.005, print_cost=False):

    w,b = initialize_param(X_train.shape[0])

    params, grads, costs = optimize(w, b, X_train, Y_train, num_iterations=num_iterations, learning_rate=learning_rate, print_cost= print_cost)

    w = params["w"]
    b = params[ "b"]

    Y_prediction_train = predict(w,b,X_train)
    Y_prediction_test = predict(w,b,X_test)

    training_accuracy = 100 - np.mean(np.abs( Y_prediction_train - Y_train) * 100)
    testing_accuracy = 100 - np.mean(np.abs( Y_prediction_test - Y_test) * 100)


    if print_cost:
        print("Training Accuracy: " + str(training_accuracy) + "%")
        print("Testing Accuracy: " + str(testing_accuracy) + "%")

    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    
    return d

def checking_test_results(index,X, Y, num_px, classes, logistic_regression_model):
    index = index
    plt.imshow(X[:, index].reshape((num_px, num_px, 3)))
    print ("y = " + str(Y[0,index]) + ", you predicted that it is a \"" + classes[int(logistic_regression_model['Y_prediction_test'][0,index])].decode("utf-8") +  "\" picture.")


def plot_cost_graph(logistic_regression_model):
    # Plot learning curve (with costs)
    costs = np.squeeze(logistic_regression_model['costs'])
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(logistic_regression_model["learning_rate"]))
    plt.show()




def predict_image(filename, w, b,num_px = 64):
    # change this to the name of your image file
    my_image = "cats/" + filename   

    # We preprocess the image to fit your algorithm.
    fname =  my_image
    image = np.array(Image.open(fname).resize((num_px, num_px)))
    plt.imshow(image)
    image = image / 255.
    image = image.reshape((1, num_px * num_px * 3)).T
    my_predicted_image = predict(w, b, image)

    print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")
    plt.show()


    





#loading the data from h5 dataset using a load function to define later
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = cats.load_h5_dataset()




def training_model(n_iterations, l_rate):
   

    #Preprocessing the Data start 
    #getting the dimensions of our matrixes
    m_train = train_set_x_orig.shape[0] #number of training examples
    m_test = test_set_x_orig.shape[0] #number of testing examples
    num_px = train_set_x_orig.shape[1] #number of pixels in the image px * px

    #flattening the data (transforming it from a matrix to a vector)
    train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T  #-1 mean multiply and flatten the rest in a 1D vector
    test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T #.T mean transpose aka like rotate the vector 

    #standarizing the dataset good for optimization apparently
    train_set_x = train_set_x_flatten / 255
    test_set_x = test_set_x_flatten / 255
    #Preprocessing the data end

    print("Logistic regression model training ....")
    Logistic_regression_model = model(train_set_x, train_set_y, test_set_x, test_set_y,num_iterations=n_iterations,learning_rate=l_rate,print_cost=True)
    print("congrats! Logistic regression model ready. \n")


    #checking_test_results(28,test_set_x, test_set_y, num_px, classes, Logistic_regression_model) 
    cats.save_results("parametersLR",  Logistic_regression_model["w"], Logistic_regression_model["b"])
    plot_cost_graph(Logistic_regression_model)
    






def trained_prediction(filename):
    
    w,b = cats.load_results("parametersLR")
    predict_image(filename, w, b)




def main():

    while True:
        try:
            choice  = input("Enter 1 for training the model \nEnter 2 for predicting results\n")
            if(choice == '1'):
                n_iterations = int(input("Enter number of iterations: "))
                l_rate = float(input("Enter the learning rate: "))
                training_model(n_iterations, l_rate)
            elif(choice == '2'):
                filename = input("Enter the file name in cats directory: ")
                trained_prediction(filename)
            else:
                print("lol....")
                return
        except:
            print("Oops something went wrong :(")

        
        

main()

