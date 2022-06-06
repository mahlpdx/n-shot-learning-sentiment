# Import Libraries
import os
import string
import re
import numpy as np
from collections import OrderedDict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from nltk import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer

# Stop words to remove from the sentences
stop_words = ['ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during', 'out',
              'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into',
              'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each', 'the',
              'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me',
              'were', 'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'while', 'above', 'both',
              'up', 'to', 'ours', 'had', 'she', 'all', 'no', 'when', 'at', 'any', 'them', 'and', 'been', 'have', 'in',
              'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over', 'why', 'so', 'can', 'did',
              'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too', 'only', 'myself', 'which',
              'those', 'i', 'after', 'few', 'whom', 'being', 'if', 'theirs', 'my', 'a', 'by', 'doing', 'it', 'how',
              'further', 'was', 'here', 'than']

# Punctuation to remove from the sentences
punctuation = string.punctuation


# Is expecting data in the form of [text, label] thus record[0] is just the text and not the label.
def sentence_cleaning(sentences_data):
    # Initialize NLP objects
    tokenizer = RegexpTokenizer(r'w+')
    stemmer = PorterStemmer()
    lemma = WordNetLemmatizer()
    # load data into numpy array for processing
    cleaned_data = np.array(sentences_data)
    # Process Each Sentence
    for sentence in cleaned_data:
        # Remove stop words
        words_list = sentence[2].split()
        new_sentence = ""
        for word in words_list:
            if word.lower() not in stop_words:
                new_sentence = new_sentence + " " + word.lower()
        sentence[2] = new_sentence
        # Remove URLS
        sentence[2] = re.sub('((www.[^s]+)|(https?://[^s]+))', ' ', sentence[2])
        # Remove punctuation
        for item in punctuation:
            sentence[2] = sentence[2].replace(item, "")
        # Tokenize Sentences
        # word_tokens = tokenizer.tokenize(sentence[0])
        # sentence
        # Apply Stemming
        stemmed_sentence = ""
        stem_list = sentence[2].split()
        for word in stem_list:
            stemmed_word = stemmer.stem(word)
            stemmed_sentence = stemmed_sentence + " " + stemmed_word
        sentence[2] = stemmed_sentence
        # Apply Lemming and add to corpus if not in it
        lemma_sentence = ""
        lemma_list = sentence[2].split()
        for word in stem_list:
            lemma_word = lemma.lemmatize(word)
            lemma_sentence = lemma_sentence + " " + lemma_word
        sentence[2] = lemma_sentence
    return cleaned_data


# vectorize the training and test data so the distances can be compared
def vectorize_dataset(sentence_training_data, sentence_test_data, to_dense=False):
    # Set up vectorizer
    vectorizer = TfidfVectorizer()
    # Separate training & test data from their labels
    training_label = np.array([row[1] for row in sentence_training_data])
    training_data = np.array([row[2] for row in sentence_training_data])
    training_agree = np.array([row[0] for row in sentence_training_data])
    test_label = np.array([row[1] for row in sentence_test_data])
    test_data = np.array([row[2] for row in sentence_test_data])
    test_agree = np.array([row[0] for row in sentence_test_data])
    # Tokenize and build corpus of all training sentences and get the vectorized results
    vectorized_train_data = vectorizer.fit_transform(training_data)
    # Vectorize the test data with the trained model
    vectorized_test_data = vectorizer.transform(test_data)
    if to_dense:
        vectorized_train_data = vectorized_train_data.todense()
        vectorized_test_data = vectorized_test_data.todense()

    # return the vectorized training and test data along with their labels.
    return vectorized_train_data.toarray(), training_label, training_agree, vectorized_test_data.toarray(), test_label, test_agree

# labels = data[:, 0]
# inputs = data[:, 1]
#
#
#
# sorted_word_count = sorted(word_counts, key=word_counts.get, reverse=True)
# # sorted_word_count = sorted( ((v,k) for k,v in word_counts.items()), reverse=True)
# digit = {}
# word_num = 1
#
# for i in range(len(sorted_word_count)):
#     if (sorted_word_count[i] not in digit.keys()):
#         digit[sorted_word_count[i]] = word_num
#         word_num += 1
# total = 0
# word_max = 500
# for i in range(len(inputs)):
#     total += len(inputs[i])
#     for j in range(len(inputs[i])):
#         num = digit.get(inputs[i][j])
#         if (num > word_max):
#             inputs[i][j] = 0
#         else:
#             inputs[i][j] = digit.get(inputs[i][j])
# ave_sentence_len = total / len(inputs)
# inputs = sequence.pad_sequences(inputs, maxlen=int(ave_sentence_len))
