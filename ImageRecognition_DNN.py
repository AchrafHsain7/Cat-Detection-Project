from cats_functions import cats
from CatNeuralNetwork import NeuralNetworkModelCat_1_3 as NNC






def main():

    choice = 0
    X_train, Y_train, X_test, Y_test, classes = cats.load_h5_dataset()

    while True:

        try:
            choice = input("Enter 1 to train model\nEnter 2 to see accuracy\nEnter 3 to predict an image\n-->")

            if choice == '1':

                X_train, X_test = NNC.normalize_data(X_train, X_test)

                layersDim = [X_train.shape[0]]
                layersDim +=[int(item) for item in input("Enter the layer dimensions : ").split()]
                learningRate = float(input("Enter the learning rate: "))
                iterations = int(input("Enter the number of iterations: "))

                print("Training.....")
                model = NNC(layersDim, learningRate, iterations, 17)
                parameters, data = model.train_model(X_train, Y_train, True)
                print("Model Able to recognize Cats succesfully:)....\n")

    
            elif choice == '2':
                print("Training Accuracy: ")
                model.accuracy(X_train, Y_train, parameters, True)
                print("Dev Accuracy: ")
                model.accuracy(X_test, Y_test, parameters, True)
                print("Testing accuracy on test sample..... \n")
                test_acc = model.test_images(parameters, classes)
                print("Test accuracy: " + str(test_acc))
                cats.plot_cost_graph(data)

            elif choice == '3':
                image = input("Enter the name of the jpg file: ")
                image =  image + ".jpg"
                model.imagePrediction(image, parameters, classes)

            else:
                print("lol....\n")
                return


        except:
            print("Oops an error occured ...\n")

main()

