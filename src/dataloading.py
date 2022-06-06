import random

sentiment_map = {'positive': 2, 'neutral': 1, 'negative': 0}


# Read in a txt file and assign class value
def read_data(file_path):
    data = []

    with open(file_path, errors='ignore') as data_file:
        lines = data_file.readlines()
        for line in lines:
            parts = line.split('\t')
            consensus = int(parts[0])
            sentiment_classification = int(parts[1])
            data.append([consensus, sentiment_classification, parts[2]])
    return data


def create_sample_data(keys, all_data):
    return [x for x in all_data if int(x[0]) in keys]


def extract_sentences(clean_data):
    sentences = []
    for row in clean_data:
        sentences.append(row[2])
    return sentences


# pull n shots from each of the given datasets
# total samples equals n * len(datasets) * num_of_classes
# pass in ignore set of sentences if need be
def get_records(n, datasets, ignore_data=[]):
    samples = []
    sample_sentences = extract_sentences(ignore_data)
    for i in range(n):
        for dataset in datasets:
            # iterate each class and find one sample to pull
            for j in range(len(sentiment_map)):
                # keep sampling points until we find one of the correct class
                k = int(random.uniform(0, len(dataset)))
                while dataset[k][1] != str(j) or dataset[k][2] in sample_sentences:
                    k = int(random.uniform(0, len(dataset)))
                samples.append(dataset[k])
                sample_sentences.append(dataset[k][2])
    return samples


# Grab test data - Take 300 test data samples from clean data
def get_test_data(clean_data):
    test_data = get_records(50, [
        create_sample_data([100], clean_data),
        create_sample_data([75], clean_data),
        create_sample_data([66], clean_data),
        create_sample_data([50], clean_data)
    ])
    return test_data
