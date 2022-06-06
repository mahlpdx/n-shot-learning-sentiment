from knn import LazyKNearestNeighbors
from dataloading import get_records, create_sample_data
from dataprocessing import vectorize_dataset


def three_class_network_compare(clean_data, constant_test_data):
    # n-shots / number of training vectors
    n_shots = [2, 6, 10]
    # 3 classes data runs
    analysis_data = []
    constant_data = get_records(10, [create_sample_data([50, 66, 75, 100], clean_data)], constant_test_data)
    for shots in n_shots:
        shots_data = []
        for sentiment in ["1", "0", "2"]:
            records = 0
            for row in constant_data:
                if records == shots: break
                if row[1] == sentiment:
                    shots_data.append(row)
                    records = records + 1
        # Embed training data
        vectorized_train_data, \
        training_label, \
        training_agree, \
        vectorized_test_data, \
        test_label, \
        test_agree = vectorize_dataset(shots_data, constant_test_data)
        max_k = shots * 2
        for k_neighs in range(1, max_k):
            print('Running KNN with 3 Classes, N-Shots: ' + str(shots) + ", and K-Neighbors: " + str(k_neighs))
            knn = LazyKNearestNeighbors(k_neighs,
                                        vectorized_train_data,
                                        training_label,
                                        training_agree)
            predictions = knn.predict(vectorized_test_data)

            analysis_data.append([3, shots, k_neighs, predictions, test_label, test_agree])

    return analysis_data


def twelve_class_network_compare(clean_data, constant_test_data):
    # n-shots / number of training vectors
    n_shots = [2, 6, 10]
    # k_neighbors - how many vectors to compare for label
    analysis_data = []
    constant_data = get_records(10, [
        create_sample_data([50], clean_data),
        create_sample_data([66], clean_data),
        create_sample_data([75], clean_data),
        create_sample_data([100], clean_data)],
                                constant_test_data)
    for shots in n_shots:
        shots_data = []
        for sentiment in ["1", "0", "2"]:
            records = 0
            for row in constant_data:
                if records == shots: break
                if row[1] == sentiment:
                    shots_data.append(row)
                    records = records + 1
        # Embed training data
        vectorized_train_data, \
        training_label, \
        training_agree, \
        vectorized_test_data, \
        test_label, \
        test_agree = vectorize_dataset(shots_data, constant_test_data)
        max_k = shots * 2
        for k_neighs in range(1, max_k):
            print('Running KNN with 12 classes, N-Shots: ' + str(shots) + ", and K-Neighbors: " + str(k_neighs))
            knn = LazyKNearestNeighbors(k_neighs,
                                        vectorized_train_data,
                                        training_label,
                                        training_agree)
            predictions = knn.predict(vectorized_test_data)
            analysis_data.append([12, shots, k_neighs, predictions, test_label, test_agree])

    return analysis_data
