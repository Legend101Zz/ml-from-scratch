import os
import sys

# 1. Find the project's root directory by going up 3 levels from this script
#    (dating_example.py -> dating-example/ -> examples/ -> knn/ -> project_root)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 2. Add the project root to Python's path
sys.path.append(project_root)
# --- End of Fix ---

from dating_example import auto_norm, file2matrix

from knn.knn import classify_knn


def dating_class_test(test_ratio=0.10):
    """
    Test the k-NN classifier on Helen's dating data.
    
    This implements the book's datingClassTest() function.
    
    Strategy:
    - Use test_ratio (e.g., 10%) of data for testing
    - Use remaining (90%) for training
    - Measure error rate
    
    Args:
        test_ratio (float): Fraction of data to use for testing
        
    Returns:
        float: Error rate (0.0 to 1.0)
    """

 

    # Load data
    dating_data, dating_labels = file2matrix('../../datasets/knn_datasets/datingTestSet2.txt')
    
    # Normalize data
    norm_data, ranges, min_vals = auto_norm(dating_data)
    
    # Calculate test set size
    m = len(norm_data)
    num_test_samples = int(m * test_ratio)
    
    print(f"Total samples: {m}")
    print(f"Test samples: {num_test_samples}")
    print(f"Training samples: {m - num_test_samples}\n")
    
    # Test using first num_test_samples as test set
    # Use remaining samples as training set
    error_count = 0
    
    for i in range(num_test_samples):
        # Test sample
        test_sample = norm_data[i]
        true_label = dating_labels[i]
        
        # Training data: everything AFTER the test set
        training_data = norm_data[num_test_samples:]
        training_labels = dating_labels[num_test_samples:]
        
        # Classify
        predicted_label, confidence = classify_knn(
            test_sample, 
            training_data, 
            training_labels, 
            k=3
        )
        
        # Print result
        result_str = "✓" if predicted_label == true_label else "✗"
        print(f"{result_str} Predicted: {predicted_label}, True: {true_label}, Confidence: {confidence:.1%}")
        
        # Count errors
        if predicted_label != true_label:
            error_count += 1
    
    # Calculate error rate
    error_rate = error_count / num_test_samples
    
    print(f"\n{'='*50}")
    print(f"Total errors: {error_count}")
    print(f"Total error rate: {error_rate:.1%}")
    print(f"Accuracy: {(1 - error_rate):.1%}")
    print(f"{'='*50}")
    
    return error_rate

def classify_person():
    """
    Interactive system for Helen to classify new potential dates.
    
    This is the book's classifyPerson() function.
    Helen inputs the three features, and we predict compatibility.
    """

    # Result mapping
    result_list = ['not at all', 'in small doses', 'in large doses']
    
    # Get input from Helen
    print("\n=== Helen's Dating Predictor ===")
    print("Enter information about the person:\n")
    
    ff_miles = float(input("Frequent flyer miles earned per year: "))
    percent_gaming = float(input("Percentage of time spent playing video games: "))
    ice_cream = float(input("Liters of ice cream consumed per year: "))
    
    # Load and normalize training data
    # Get the directory where this script (dating_example.py) is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Join that directory path with the filename
    # This creates a reliable, absolute path to our data file
    file_path = os.path.join(script_dir, 'datingTestSet2.txt')

    dating_data, dating_labels = file2matrix(file_path)
    norm_data, ranges, min_vals = auto_norm(dating_data)
    
    # Normalize the input using TRAINING data parameters
    # This is critical - we use the same normalization as training!
    input_array = [ff_miles, percent_gaming, ice_cream]
    normalized_input = []
    for j in range(len(input_array)):
        if ranges[j] == 0:
            normalized_input.append(0.5)
        else:
            normalized_input.append((input_array[j] - min_vals[j]) / ranges[j])
    
    # Classify
    prediction, confidence = classify_knn(
        normalized_input,
        norm_data,
        dating_labels,
        k=3
    )
    
    # Display result
    print(f"\n{'='*50}")
    print(f"You will probably like this person: {result_list[prediction - 1]}")
    print(f"Confidence: {confidence:.1%}")
    print(f"{'='*50}\n")
    
classify_person()