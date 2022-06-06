import numpy as np
import operator
from distancefunctions import cosine_distance


class LazyKNearestNeighbors:

    def __init__(self,
                 k_neighbors,
                 vectorized_train_data,
                 training_label,
                 training_agree):
        self.k = k_neighbors
        self.distance_function = cosine_distance

        # TODO add vectorization here
        self.neighbor_vectors = vectorized_train_data
        self.neighbor_labels = training_label
        self.neighbor_agree = training_agree

    def predict(self, query_vectors):
        # Label Predictions
        label_predictions = []

        # Loop over vector values
        for query_vector in query_vectors:
            # calculate the distance between k neighbors
            distances = np.array(
                [self.distance_function(query_vector, neighbor) for neighbor in self.neighbor_vectors])

            # return sorted distance for closest neighbors
            distances_sorted = distances.argsort()[:self.k]

            # get neighbor class counts
            neighbor_count = {}

            # for each neighbor add its label via the index
            for idx in distances_sorted:
                if self.neighbor_labels[idx] in neighbor_count:
                    neighbor_count[self.neighbor_labels[idx]] += 1
                else:
                    neighbor_count[self.neighbor_labels[idx]] = 1

            # get the most frequent class label
            neighbor_count = sorted(neighbor_count.items(), key=operator.itemgetter(1), reverse=True)
            # add to predictions and return
            label_predictions.append(neighbor_count[0][0])
        return label_predictions
