import os
from preprocess import preprocess_data
from classifier import NaiveBayesClassifier

def main():
    # Read the data from the file
    baseDir = os.path.dirname(os.path.abspath(__file__))
    dataDir = os.path.join(baseDir, 'data')

    trainFeaturePath = os.path.join(dataDir, 'traindata.txt')
    trainLabelPath = os.path.join(dataDir, 'trainlabels.txt')
    testFeaturePath = os.path.join(dataDir, 'testdata.txt')
    testLabelPath = os.path.join(dataDir, 'testlabels.txt')
    stoplistPath = os.path.join(dataDir, 'stoplist.txt')

    # Pre-process data
    X_train, y_train, vocabulary = preprocess_data(trainFeaturePath, trainLabelPath, stoplistPath)
    X_test, y_test, _ = preprocess_data(testFeaturePath, testLabelPath, stoplistPath, vocabulary=vocabulary)

    # Initialize the classifier and train the training data
    classifier = NaiveBayesClassifier(len(vocabulary))
    classifier.fit(X_train, y_train)

    # calculate the training and testing accuracy
    trainAccuracy = classifier.accuracy(X_train, y_train)
    testAccuracy = classifier.accuracy(X_test, y_test)

    print(f"Training Accuracy: {trainAccuracy * 100:.2f}%")
    print(f"Testing Accuracy: {testAccuracy * 100:.2f}%")

if __name__ == '__main__':
    main()