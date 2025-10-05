from collections import Counter

from preprocessing import normalize_features

from ..foundations.distance_metrics import (euclidean_distance,
                                            manhattan_distance)


def classify_knn(unknown_point,training_data, training_labels, k=3, distance_metric='euclidean'):
    """ 
    Classify a single data point using k-Nearest Neighbors algorithm.
    
    This is the heart of k-NN. We're implementing the exact algorithm described
    in the book (ML in action), but using the mathematical primitives we built ourselves.
    
    Algorithm steps (as described in the book):
    1. Calculate distance between unknown point and every training point
    2. Sort distances in increasing order
    3. Take k items with lowest distances
    4. Find majority class among these k neighbors
    5. Return majority class as prediction
    
    Args:
        unknown_point (list): The data point we want to classify
        training_data (list of lists): All training examples
        training_labels (list): Class labels for training data
        k (int): Number of neighbors to consider
        distance_metric (str): Which distance function to use
        
    Returns:
        predicted_class: The predicted class for unknown_point
    """
    if len(training_data) != len(training_labels):
        raise ValueError("Training data and labels must have the same length")
    
    if k > len(training_data):
        raise ValueError(f"k ({k}) cannot be larger than number of training examples ({len(training_data)})")
    
    # Step 1: Calculate distances to all training points
    distances = []
    for i, training_point in enumerate(training_data):
        if distance_metric == 'euclidean':
            dist = euclidean_distance(unknown_point,training_point)
        else: 
            dist = manhattan_distance(unknown_point,training_point)
    
        distances.append((dist,training_labels[i]))
        
    # Step 2 : Sort by distance (smallest to largest)
    distances.sort(key=lambda x:x[0])
    
    # Step 3 : Get the k nearest neighbors
    k_nearest = distances[:k]
    
    # Step 4 : Extract just the labels from k nearest neighbours
    k_nearest_labels = [label for (dist, label) in k_nearest]
    
    # Step 5 : Find the majority class using Counter
    vote_count = Counter(k_nearest_labels)
    predicted_class = vote_count.most_common(1)[0][0]
    
    # Calculate confidence (percentage of neighbors that voted for winner)
    confidence = vote_count.most_common(1)[0][1] / k
    
    return predicted_class, confidence

def knn_with_probabilities(unknown_point, training_data, training_labels, k=3):
    """
    Enhanced k-NN that returns class probabilities instead of just predictions.
    
    Instead of just returning the majority class, we return the probability of each class
    based on the voting results.
    
    If 3 out of 5 neighbors are class 'A', then P(class='A') = 0.6
    
    This gives us a measure of confidence in our prediction and allows for
    more nuanced decision-making.
    """
    if k > len(training_data):
        raise ValueError(f"k ({k}) cannot exceed training set size")
    
    # Calculate distances
    distances = []
    for i, training_point in enumerate(training_data):
        dist = euclidean_distance(unknown_point, training_point)
        distances.append((dist, training_labels[i]))
    
    # Sort and get k nearest
    distances.sort(key=lambda x: x[0])
    k_nearest_labels = [label for (dist, label) in distances[:k]]
    
    # Calculate probabilities for each class
    vote_count = Counter(k_nearest_labels)
    probabilities = {}
    
    for class_label, count in vote_count.items():
        # P(class) = count of class in k neighbors / k
        probabilities[class_label] = count / k
    
    return probabilities