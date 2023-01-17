import numpy as np
import h5py
import matplotlib.pyplot as plt



class cats:

    def load_h5_dataset():
        hf_train = h5py.File("train_catvnoncat.h5", 'r')
        hf_test = h5py.File("test_catvnoncat.h5", 'r')



        trainX = hf_train.get('train_set_x')
        trainY = hf_train.get('train_set_y')
        testX = hf_test.get('test_set_x')
        testY = hf_test.get('test_set_y')
        classes = hf_train.get('list_classes')


        train_set_x = np.array(trainX)
        train_set_y = np.array(trainY)
        test_set_x = np.array(testX)
        test_set_y = np.array(testY)
        classes = np.array(classes)

        return train_set_x, train_set_y, test_set_x, test_set_y, classes


    def save_results(filename, w,b):
        np.savez(filename, w=w, b=b)
        
    def load_results(filename):
        fileN = filename+".npz"
        data = np.load(fileN)
        w = data["w"]
        b = data["b"]

    def save_results2(filename, parameters):
        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]

        np.savez(filename, W1=W1, b1=b1, W2=W2, b2=b2)

        
        
    def load_results2(filename):
        fileN = filename+".npz"
        data = np.load(fileN)
        W1 = data["W1"]
        b1 = data["b1"]
        W2 = data["W2"]
        b2 = data["b2"]

        parameters = {"W1":W1,
                  "b1": b1,
                  "W2": W2,
                  "b2":b2}

        return parameters


    def plot_cost_graph(model):
        # Plot learning curve (with costs)
        costs = np.squeeze(model['costs'])
        plt.plot(costs)
        plt.ylabel('cost')
        plt.xlabel('iterations (per hundreds)')
        plt.title("Learning rate =" + str(model["learning_rate"]))
        plt.show()









        




        





        

