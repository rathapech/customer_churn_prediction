"""
    This script aims at predicting the customer churn based on IBM dataset available at https://www.kaggle.com/blastchar/telco-customer-churn     
"""

import random
from collections import Counter
import numpy as np
from numpy import genfromtxt
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class ReadContent(object):
    """"
        This is the main class
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def load_content(self):
        """ Below scripts are to get content from all files in each folders """
        

        # Modify the codes at below line to where the data is kept 
        my_data = genfromtxt('../dataset/churn_cleaned_new.csv', delimiter=',')
        my_data = np.asarray(my_data)
        return my_data
    
    def training_SVM(self, training_data, training_label):
        """" Training SVM for classification """
        clf_svm = svm.SVC(kernel = 'rbf', C = 10, gamma = 0.01)
        #clf_svm = svm.SVC()
        clf_svm.fit(training_data, training_label)

        return clf_svm
    
    def training_MLP(self, training_data, training_label):
        """" Training Multi Layer Perceptron for classification """
        clf_nn = MLPClassifier(activation= 'logistic', solver='lbfgs', alpha = 0.1, momentum = 0.1, hidden_layer_sizes=(350, 350, 250))
        clf_nn.fit(training_data, training_label)

        return clf_nn

    def training_NB(self, training_data, training_label):
        """" Training Naive Bayest for classification """
        #clf_gnb = BernoulliNB()
        #clf_gnb = MultinomialNB()
        clf_gnb = GaussianNB()
        clf_gnb.fit(training_data, training_label)

        return clf_gnb

    def predicting_model(self, model, testing_data):
        """ Testing model """
        predicted_labels = model.predict(testing_data)

        return predicted_labels

    def data_division(self, data, num_each_class_4_test):
        """This method aims at opening file for division"""

        # Shuffle records for randomly selecting the testing records 

        start_range = 1868 # Number of records labeled as churn 
        end_range = len(data)
        
        random_fraud_idx = random.sample(range(0, 1868), 1868) 
        random_non_fraud_idx = random.sample(range(start_range, end_range-1), end_range-1-start_range)

        # Combining the random data from two classes 
        new_idx = random_fraud_idx+random_non_fraud_idx

        # Splitting the feature variables and labels 
        features = np.array(data[new_idx, 0:29])
        labels = np.array(data[new_idx, 30]) 

        unique_label = list(set(labels))
        unique_label = sorted(unique_label, reverse=True)

        label_data = Counter(labels)

        temp0 = random.sample(range(0, label_data[1]), num_each_class_4_test)
        temp1 = random.sample(range(label_data[1], label_data[0]+label_data[1]), num_each_class_4_test)

        all_testing = temp0 + temp1

        testing_data = features[all_testing]
        testing_labels = labels[all_testing]

        # Removing the testing data from all the data since testing records are used in predicting only, not in training
        training_data = np.delete(features, all_testing, 0)
        training_labels = np.delete(labels, all_testing)

        return training_data, training_labels, testing_data, testing_labels


if __name__ == "__main__":

    # Number of records from each class to be used as testing data
    # Adjusting this variable to test the robustness of the method 
    NUM_TEST_RECORDS = 100

    # Loading the data to work 
    TEST = ReadContent()
    RECORDS = TEST.load_content()

    # Number of independent simulations
    NUM_RUN = 5

    # Allocating memory for some variables 
    TOTAL_SCORE_SVM = []
    TOTAL_SCORE_MLP = []
    TOTAL_SCORE_NB = []

    PRE_NB =[]
    REC_NB = []
    F_SCORE_NB = []

    PRE_MLP = []
    REC_MLP = []
    F_SCORE_MLP = []

    PRE_SVM = []
    REC_SVM = []
    F_SCORE_SVM = []

    for i in range(NUM_RUN):
        print("\n==================", i+1, "================\n")

        TRAINING_DATA, TRAINING_LABELS, TESTING_DATA, TESTING_LABELS = \
        TEST.data_division(RECORDS, NUM_TEST_RECORDS)

        scaler = StandardScaler()
        scaler.fit(TRAINING_DATA)
        TRAINING_DATA = scaler.transform(TRAINING_DATA)
        TESTING_DATA = scaler.transform(TESTING_DATA)

        # Reduce the dimensions by using Principal Component Analysis (PCA)
        pca = PCA(n_components = 15)
        pca.fit(TRAINING_DATA)
        TRAINING_DATA = pca.transform(TRAINING_DATA)
        TESTING_DATA = pca.transform(TESTING_DATA)

        # Training the models by using three different methods, e.g., SVM, MLP, and Naive Bayes  
        MODEL_SVM = TEST.training_SVM(TRAINING_DATA, TRAINING_LABELS)
        MODEL_MLP = TEST.training_MLP(TRAINING_DATA, TRAINING_LABELS)
        MODEL_NB = TEST.training_NB(TRAINING_DATA, TRAINING_LABELS)

        # Predicting the data based on the trained model
        PREDICTED_LABELS_SVM = TEST.predicting_model(MODEL_SVM, TESTING_DATA)
        PREDICTED_LABELS_MLP = TEST.predicting_model(MODEL_MLP, TESTING_DATA)
        PREDICTED_LABELS_NB = TEST.predicting_model(MODEL_NB, TESTING_DATA)

        # ===================================================================#

        # Computing the performance from each method
        # SMV
        SCORE_SVM = accuracy_score(TESTING_LABELS, PREDICTED_LABELS_SVM)
        score_svm = precision_recall_fscore_support(TESTING_LABELS, PREDICTED_LABELS_SVM, average='binary')
        PRE_SVM.append(score_svm[0])
        REC_SVM.append(score_svm[1])
        F_SCORE_SVM.append(score_svm[2])
        print('SVM accuracy: ', SCORE_SVM*100)
        TOTAL_SCORE_SVM.append(SCORE_SVM)

        
        # MPL
        SCORE_MLP = accuracy_score(TESTING_LABELS, PREDICTED_LABELS_MLP)
        TOTAL_SCORE_MLP.append(SCORE_MLP)
        score_mlp = precision_recall_fscore_support(TESTING_LABELS, PREDICTED_LABELS_MLP, average='binary')
        PRE_MLP.append(score_mlp[0])
        REC_MLP.append(score_mlp[1])
        F_SCORE_MLP.append(score_mlp[2])
        #PRE_REC_FSCOREpre_mlpT_SCORE)
        print('MLP accuracy: ', SCORE_MLP*100)
        # print('Predicted accuracy from MLP: ', SCORE*100, '%')
    
        
        # NB
        SCORE_NB = accuracy_score(TESTING_LABELS, PREDICTED_LABELS_NB)
        TOTAL_SCORE_NB.append(SCORE_NB)
        score_nb = precision_recall_fscore_support(TESTING_LABELS, PREDICTED_LABELS_NB, average='binary')
        PRE_NB.append(score_nb[0])
        REC_NB.append(score_nb[1])
        F_SCORE_NB.append(score_nb[2])
        #PRE_REC_FSCORE_MLP.append(T_SCORE_NB)
        print('NB accuracy: ', SCORE_NB*100)
    
    # Display all the performane measured by accuracy, precision, recall and F-Score
    print("===========================================\n")
    print('Accuracy\n')
    print(np.average(TOTAL_SCORE_SVM))
    print(np.std(TOTAL_SCORE_SVM))

    print(np.average(TOTAL_SCORE_MLP))
    print(np.std(TOTAL_SCORE_MLP))

    print(np.average(TOTAL_SCORE_NB))
    print(np.std(TOTAL_SCORE_NB))
    print('\n')

    print('Precision \n')
    print(np.average(PRE_SVM))
    print(np.std(PRE_SVM))

    print(np.average(PRE_MLP))
    print(np.std(PRE_MLP))

    print(np.average(PRE_NB))
    print(np.std(PRE_NB))
    print('\n')

    print('Recall \n')
    print(np.average(REC_SVM))
    print(np.std(REC_SVM))

    print(np.average(REC_MLP))
    print(np.std(REC_MLP))

    print(np.average(REC_NB))
    print(np.std(REC_NB))
    print('\n')

    print('F Score \n')
    print(np.average(F_SCORE_SVM))
    print(np.std(F_SCORE_SVM))

    print(np.average(F_SCORE_MLP))
    print(np.std(F_SCORE_MLP))

    print(np.average(F_SCORE_NB))
    print(np.std(F_SCORE_NB))

    LABELS_1 = (TESTING_LABELS > 0).sum()
    print('\nNumer of frauds in total testing records: ', LABELS_1)
    LABELS_0 = (TESTING_LABELS == 0).sum()
    print('Numer of usual transactions in total testing records: ', LABELS_0)
    print("===========================================\n")
