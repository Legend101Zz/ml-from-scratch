import os
import sys

# 1. Find the project's root directory by going up 3 levels from this script
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 2. Add the project root to Python's path
sys.path.append(project_root)
from knn.knn import classify_knn


def img2vector(filename):
    """
    Convert a 32x32 text image file into a 1024-element vector.
    
    This is the book's img2vector() function.
    
    The image is 32x32 = 1024 pixels. We read each pixel and put it
    in a flat vector [pixel1, pixel2, ..., pixel1024].
    
    Args:
        filename (str): Path to text file containing 32x32 image
        
    Returns:
        list: 1024-element vector of 0s and 1s
    """
    return_vector = []
    
    with open(filename, 'r') as fr:
        # Read 32 lines
        for i in range(32):
            line_str = fr.readline()
            
            # Read 32 characters from each line
            for j in range(32):
                # Convert character to integer and append
                return_vector.append(int(line_str[j]))
    
    return return_vector


def visualize_digit(filename):
    """
    Display what a digit looks like from the text file.
    """
    with open(filename, 'r') as fr:
        print(f"\nDigit from file: {filename}\n")
        for i in range(32):
            line = fr.readline().strip()
            # Replace 1s with█ and 0s with space for better visualization
            visual_line = line.replace('1', '█').replace('0', ' ')
            print(visual_line)
    print()
    

def handwriting_class_test():
    """
    Test k-NN on handwritten digit recognition.
    
    This is the book's handwritingClassTest() function.
    
    Process:
    1. Load all training images and labels
    2. Convert each to a vector
    3. For each test image:
       - Convert to vector
       - Classify using k-NN
       - Compare to true label
    4. Calculate error rate
    """

    # Load training data
    hw_labels = []
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # FIX: Use os.listdir() to get the list of files
    training_dir = os.path.join(script_dir, './trainingDigits')
    training_file_list = os.listdir(training_dir)  # This returns a LIST of filenames
    
    m = len(training_file_list)
    training_matrix = []
    
    print(f"Loading {m} training samples...")
    
    for i, filename in enumerate(training_file_list):
        # Skip hidden files like .DS_Store on Mac
        if filename.startswith('.'):
            continue
            
        if i % 500 == 0:
            print(f"  Loaded {i}/{m} training samples...")
        
        # Extract class label from filename
        # Filename format: "digit_instance.txt" e.g., "0_13.txt"
        file_str = filename.split('.')[0]  # Remove .txt
        class_num = int(file_str.split('_')[0])  # Get digit before underscore
        
        hw_labels.append(class_num)
        
        # Convert image to vector
        img_path = os.path.join(training_dir, filename)  # Use training_dir here
        img_vector = img2vector(img_path)
        training_matrix.append(img_vector)
    
    print(f"Training data loaded: {len(training_matrix)} samples\n")
    
    # Test the classifier
    # FIX: Use os.listdir() here too
    test_dir = os.path.join(script_dir, './testDigits')
    test_file_list = os.listdir(test_dir)  # This returns a LIST of filenames
    
    error_count = 0
    m_test = len(test_file_list)
    
    print(f"Testing on {m_test} samples...")
    print(f"{'='*60}\n")
    
    for i, filename in enumerate(test_file_list):
        # Skip hidden files
        if filename.startswith('.'):
            continue
            
        # Extract true class
        file_str = filename.split('.')[0]
        true_class = int(file_str.split('_')[0])
        
        # Convert test image to vector
        img_path = os.path.join(test_dir, filename)  # Use test_dir here
        test_vector = img2vector(img_path)
        
        # Classify with k=3
        predicted_class, confidence = classify_knn(
            test_vector,
            training_matrix,
            hw_labels,
            k=3
        )
        
        # Check result
        if predicted_class != true_class:
            error_count += 1
            result = "✗ ERROR"
        else:
            result = "✓"
        
        # Print every 50th result plus all errors
        if i % 50 == 0 or predicted_class != true_class:
            print(f"{result} | File: {filename:15} | Predicted: {predicted_class} | True: {true_class} | Conf: {confidence:.0%}")
    
    # Results
    error_rate = error_count / m_test
    
    print(f"\n{'='*60}")
    print(f"Results:")
    print(f"  Total test samples: {m_test}")
    print(f"  Total errors: {error_count}")
    print(f"  Error rate: {error_rate:.1%}")
    print(f"  Accuracy: {(1-error_rate):.1%}")
    print(f"{'='*60}")
    
    return error_rate


# Test it
if __name__ == "__main__":
    # Get the directory where this script (dating_example.py) is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Join that directory path with the filename
    # This creates a reliable, absolute path to our data file
    file_path = os.path.join(script_dir, './trainingDigits/0_105.txt')

    # Visualize a digit
    visualize_digit(file_path)
    
    # Convert to vector
    vector = img2vector(file_path)
    print(f"Vector length: {len(vector)}")
    print(f"First 32 elements: {vector[:32]}")
    print(f"Vector values: {set(vector)}")  # Should be {0, 1}
    
    # Run the test
    handwriting_class_test()