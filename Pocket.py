
# Hannah Keefe
# 20053849
# Training perceptron for the Iris Data, implementing Pocket algorithm

import numpy as np
from numpy import random
from csv import reader
import seaborn as sns
import matplotlib.pyplot as plt

# class for the "pocket" object
class BestPocket:
     def __init__(self):
        self.best_weights = np.zeros(4+ 1)
        self.misclassify_count = -1

# class to implement NN with pocket alg
class Pocket:
    def __init__(self, l_rate, num_epochs, threshold):
        # learning rate set by user
        self.l_rate = l_rate
        # number of iterations set by user
        self.num_epochs = num_epochs
        # threshold value set by user
        self.threshold = threshold
        self.errors = []
        # create a pocket object
        self.pocket = BestPocket()
        # keeps class of misclassify_counts for diffrent weights
        self.misclassify_record = []
        # 3 weight vectors because 3 outputs,5 elements in each vector
        # weight is bounded by 5
        self.weights = 5*random.rand(3,5)
        # read in training and test data
        self.training_data = self.file_read('iris_train.txt')
        self.test_data = self.file_read('iris_test.txt')

    # check class label and return numeric representation of class
    def label(self, row):
        species = []
        if(row == 'Iris-setosa'):
            species = [1,0,0]
        elif(row == 'Iris-versicolor'):
            species = [1,1,0]
        elif(row == 'Iris-virginica'):
            species = [1,1,1]
        return species

    # calculate activation to see if perceptron "fires"
    def activation(self, y):
        output = []
        for elem in y:
            if (elem > self.threshold): #and x == np.amax(z)):
                output.append(1)
            else:
                output.append(0)
        return np.array(output)

    # given label covert a species to one sigle int identifier for the matrix
    def label_to_int(self, label):
        if (label[0] == 1 and label[1] == 0 and label[2] == 0):
            return 0
        elif(label[0] == 1 and label[1] == 1 and label[2] == 0):
            return 1
        elif(label[0] == 1 and label[1] == 1 and label[2] == 1):
            return 2

    # given species string convert species to a single int identifier for the matrix
    def String_to_int(self, species):
        if species == 'Iris-setosa':
            return 0
        elif species == 'Iris-versicolor':
            return 1
        elif species == 'Iris-virginica':
            return 2

    # train the Neural Network
    def train(self, training_data):
        # iterate through epochs
        for _ in range (self.num_epochs):
            # initalze errors to 0
            errors = 0
            for row in training_data:
                # get the feature data
                x = [1,float(row[0]), float(row[1]), float(row[2]), float(row[3])]
                # output of dot product to get total activation
                y = np.dot(self.weights, x)
                # check is activation exceeds threshold
                output = self.activation(y)

                #expected output and predicted output
                type = row[4]
                expected_output = []
                expected_ouput = self.label(type)

                for j in range(3):
                    # false negative
                    if(expected_ouput[j] == 1 and output[j] == 0): #If the output should be 1...
                        errors += 1
                        self.weights[j] += np.multiply (self.l_rate, x)
                    # false positive
                    elif(expected_ouput[j] == 0 and output[j] == 1): #If the output shouldn't be 1...
                        errors += 1
                        self.weights[j] -= np.multiply (self.l_rate, x)

            # check to see if we need to replace the pocket
            if(self.pocket.misclassify_count == -1 or self.pocket.misclassify_count > errors or errors == 0):
                # update the best weight vectors
                self.pocket.best_weights = self.weights
                self.pocket.misclassify_count = errors

            # no errors have found the perfect weight vector
            if (errors== 0):
                break

            # update the misclassify count each epoch
            self.misclassify_record.append(self.pocket.misclassify_count)
        return self

    def test(self,test_data):
        err = 0
        correct = 0
        false_positives = 0
        false_negatives = 0
        true_postives = 0
        #initialize confusion matrix values to 0
        matrix = np.zeros((3, 3))

        for row in test_data:
            # get the feature data
            x = [1,float(row[0]), float(row[1]), float(row[2]), float(row[3])]
            # output of dot product to get total activation
            y = np.dot(self.pocket.best_weights, x)

            #expected output and predicted output
            type = row[4]
            expected_output = []
            expected_ouput = self.label(type)
            output = self.activation(y)

            # add maytrix data
            matrix[(self.label_to_int(output)), self.String_to_int(row[4])] += 1

            # classify errors to calculate Recall and Precision
            for j in range(3):
                # false negative
                if(expected_ouput[j] == 1 and output[j] == 0): #If the output should be 1...
                    err += 1
                    false_negatives += 1
                # false positive
                elif(expected_ouput[j] == 0 and output[j] == 1): #If the output shouldn't be 1...
                    err += 1
                    false_positives += 1
                # true positives
                elif(expected_ouput[j] == 1 and output[j] == 1): #If the output shouldn't be 1...
                    true_postives += 1

        accuracy = (len(test_data) - err)/(len(test_data)) * 100

        # check to make sure > 0 cause if = 0 error
        if ((true_postives + false_positives) >0):
            precision = (true_postives/(true_postives + false_positives))
        else:
            precision = 0
        # check to make sure > 0 cause if not error
        if (true_postives + false_negatives >0):
            recall = (true_postives/(true_postives + false_negatives))
        else:
            precision = 0

        print("The accuracy is: ",accuracy, "%")
        print("The precision is: ",precision)
        print("The recall is:", recall)
        sns.heatmap(matrix, annot=True, xticklabels=['setosa', 'versicolor', 'virginica'], yticklabels= ['setosa', 'versicolor', 'virginica'], cmap='Reds')
        plt.ylabel('predicted class')
        plt.xlabel('actual class')
        plt.show()

    #function to read the training and test data
    def file_read(self,fileName):
        info = list()
        with open(fileName, 'r') as file:
            f = reader(file)
            for row in f:
                info.append(row)
        return info


percept = Pocket(0.1, 300, 10)
percept.train(percept.training_data)
percept.test(percept.test_data)
