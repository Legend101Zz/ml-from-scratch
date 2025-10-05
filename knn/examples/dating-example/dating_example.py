""" 
The Problem:

Helen is using a dating website and has been matched with many people, but she realizes she doesn't like everyone equally. After introspection, she categorizes people into three groups:

Class 1: People she didn't like at all
Class 2: People she liked in small doses (weekday hangouts)
Class 3: People she liked in large doses (weekend hangouts)

Helen collected data on 1000 people with three features she believes are useful:

Frequent flyer miles per year (how much they travel)
Percentage of time spent playing video games
Liters of ice cream consumed per week

Let's help Helen predict if she'll like future matches!
"""

import os
import sys

# 1. Find the project's root directory by going up 3 levels from this script
#    (dating_example.py -> dating-example/ -> examples/ -> knn/ -> project_root)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 2. Add the project root to Python's path
sys.path.append(project_root)
# --- End of Fix ---

# 3. Now we can use a clean, absolute import from the project root
from foundations.distance_metrics import euclidean_distance


# Okay so,First we need to define the function to load the data from the text file
def file2matrix(filename):
    """ 
    Parse the dating dataset from the text file.
    
    The book's approach is to read the file twice (once to count the lines and once to read data) .. we'll try to improve that and read all at once.
    
    File format: Each line has 4 tab-seperated values:
   - frequent flyer miles
    - percentage time gaming  
    - liters of ice cream
    - class label (1, 2, or 3)
    
    Example line: "40920  8.326976  0.953952  3"
    
    Args:
        filename (str): Path to the data file
        
    Returns:
        tuple: (feature_matrix, class_labels)
               feature_matrix is list of [miles, gaming, ice_cream]
               class_labels is list of integers (1, 2, or 3)
    """
    # Read all the lines first
    with open(filename, 'r') as fr:
        lines = fr.readlines()
        
    # Prepare data structures
    feature_matrix = []
    class_labels = []
    
    # Parse each line
    for line in lines:
        line = line.strip() # Remove the whitespace and newline character.
        
        # Split by tabs
        values = line.split('\t')
        
        # First 3 values are features and last in the class label
        features = [float(values[0]), float(values[1]), float(values[2])]
        feature_matrix.append(features)
        
        # Class label is the last value
        class_labels.append(int(values[3]))
        
    return feature_matrix, class_labels


def auto_norm(dataset):
    """
    Normalize features to 0-1 range using min-max scaling.
    
    This is the book's autoNorm() function implemented from scratch.
    
    Formula: normalized_value = (value - min) / (max - min)
    
    This ensures all features contribute equally to distance calculations.
    
    Args:
        dataset (list of lists): Each inner list is a sample with features
        
    Returns:
        tuple: (normalized_dataset, ranges, min_values)
               We return ranges and min_values so we can normalize test data
               using the SAME parameters (to avoid data leakage!)
    """
    # Get number of features
    n_samples = len(dataset)
    n_features = len(dataset[0])
    
    # Find min and max for each feature
    min_vals = []
    max_vals = []
    
    for j in range(n_features):
        # Extract jth feature from all samples
        feature_column = [dataset[i][j] for i in range(n_samples)]
        min_vals.append(min(feature_column))
        max_vals.append(max(feature_column))
    
    # Calculate ranges
    ranges = [max_vals[j] - min_vals[j] for j in range(n_features)]
    
    # Normalize the dataset
    normalized_dataset = []
    for i in range(n_samples):
        normalized_sample = []
        for j in range(n_features):
            if ranges[j] == 0:  # Avoid division by zero
                normalized_sample.append(0.5)
            else:
                normalized_value = (dataset[i][j] - min_vals[j]) / ranges[j]
                normalized_sample.append(normalized_value)
        normalized_dataset.append(normalized_sample)
    
    return normalized_dataset, ranges, min_vals


def demonstrate_normalization_effect():
    """
    Show the dramatic difference normalization makes.
    """
    # Example from the book
    person_3 = [0, 20000, 1.1]
    person_4 = [67, 32000, 0.1]

    
    print("WITHOUT normalization:")
    dist_raw = euclidean_distance(person_3, person_4)
    print(f"Distance: {dist_raw:.2f}")
    print(f"  (67-0)² = {(67-0)**2}")
    print(f"  (32000-20000)² = {(32000-20000)**2}")
    print(f"  (0.1-1.1)² = {(0.1-1.1)**2}")
    print(f"  Notice: Flyer miles term dominates completely!\n")
    
    # Normalize using dataset stats
    dataset = [person_3, person_4]
    norm_dataset, ranges, mins = auto_norm(dataset)
    
    print("WITH normalization:")
    dist_norm = euclidean_distance(norm_dataset[0], norm_dataset[1])
    print(f"Distance: {dist_norm:.2f}")
    print(f"  Person 3 normalized: {norm_dataset[0]}")
    print(f"  Person 4 normalized: {norm_dataset[1]}")
    print(f"  Now all features contribute fairly!")



# Get the directory where this script (dating_example.py) is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Join that directory path with the filename
# This creates a reliable, absolute path to our data file
file_path = os.path.join(script_dir, 'datingTestSet2.txt')

dating_data, dating_labels = file2matrix(file_path)

print(f"Loaded {len(dating_data)} samples")
print(f"First sample: {dating_data[0]}")
print(f"First label: {dating_labels[0]}")
print(f"\nFirst 20 labels: {dating_labels[:20]}")

demonstrate_normalization_effect()

from visualisation import visualize_dating_data

visualize_dating_data(dating_data, dating_labels )
