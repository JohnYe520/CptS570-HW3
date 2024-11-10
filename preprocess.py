import os

# Read the data from the file
def load_data(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    processedLine = []

    # process each line by removing the whitespaces and make them to lower case
    for line in lines:
        processedLine.append(line.strip().lower())

    return processedLine

# Read the stopwords from the file
def load_stopwords(stoplist_path):
    with open(stoplist_path, 'r') as file:
        lines = file.readlines()

    processedStopwords = set()
    # process each line by removing the whitespaces and make them to lower case
    for line in lines:
        processedStopwords.add(line.strip().lower())

    return processedStopwords

# Get vocabulary and keep them in alphabetical order
def create_vocabulary(data, stopwords):
    vocab = set()
    # split each message into words, and add them to the vocabulary if they are not a stopword
    for message in data:
        words = message.split()
        for word in words:
            if word not in stopwords:
                vocab.add(word)
    
    sortedVocab = sorted(vocab)
    return sortedVocab

# Convert the message to feature vector 
def message_to_feature_vector(message, vocabulary):
    words = set(message.split())
    featureVector = []
    # check whether the word is in the message or not
    for vocabWord in vocabulary:
        if vocabWord in words:
            featureVector.append(1)
        else:
            featureVector.append(0)
    return featureVector

# Preprocess the data
def preprocess_data(data_path, labels_path, stoplist_path, vocabulary=None):
    data = load_data(data_path)
    labels = list(map(int, load_data(labels_path)))
    stopwords = load_stopwords(stoplist_path)

    if vocabulary is None:
        vocabulary = create_vocabulary(data, stopwords)
    
    featureVectors = []
    for message in data:
        featureVector = message_to_feature_vector(message, vocabulary)
        featureVectors.append(featureVector)
    
    return featureVectors, labels, vocabulary