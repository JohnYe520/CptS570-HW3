import numpy as np

class NaiveBayesClassifier:
    # initialize the classifier
    def __init__(self, vocabularySize):
        self.vocabularySize = vocabularySize
        self.wordCount = [np.zeros(vocabularySize), np.zeros(vocabularySize)]
        self.count = [0, 0]
        self.priors = [0, 0]

    # train the classifier by calculating the word count and class priors
    def fit(self, X, y):
        for featureVector, label in zip(X, y):
            self.count[label] += 1
            self.wordCount[label] += featureVector

        total_samples = len(y)
        # calculate the class prior
        self.priors[0] = self.count[0] / total_samples
        self.priors[1] = self.count[1] / total_samples

    # make precitions for a given feature vector
    def predict(self, featureVector):
        # initialize the log probabilities for each class
        logProb0 = np.log(self.priors[0])
        logProb1 = np.log(self.priors[1])

        # calculate the log probabilities for each word in the vocabulary
        for i in range(self.vocabularySize):
            # Ensure featureVector has correct length
            if i >= len(featureVector):
                raise IndexError(f"Feature vector length ({len(featureVector)}) is less than vocabulary size ({self.vocabularySize}).")
            # Calculate conditional probability of each word in class 0 using Laplace Smoothing
            wordProb0 = (self.wordCount[0][i] + 1) / (self.count[0] + self.vocabularySize)
            logWordProb0 = featureVector[i] * np.log(wordProb0)
            # Calculate conditional probability of each word in class 1 using Laplace Smoothing
            wordProb1 = (self.wordCount[1][i] + 1) / (self.count[1] + self.vocabularySize)
            logWordProb1 = featureVector[i] * np.log(wordProb1)

            logProb0 += logWordProb0
            logProb1 += logWordProb1

        # make prediction on the class based on the probability
        if  logProb0 > logProb1:
            return 0
        else:
            return 1
    
    # calculate the accuracy for the predictions
    def accuracy(self, X, y):
        predictions = []

        for featureVector in X:
            prediction = self.predict(featureVector)
            predictions.append(prediction)
        
        trueLabel = 0
        for pred, true in zip(predictions, y):
            if pred == true:
                trueLabel += 1
        accuracy = trueLabel / len(y)

        return accuracy