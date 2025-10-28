"""
NORMALIZATION (MIN-MAX SCALING): BOUNDING FEATURES TO A RANGE
=============================================================

Normalization scales each feature to a specific range, typically [0, 1] or [-1, 1].
Unlike standardization (which uses mean and std), normalization uses min and max values.

TEACHING PHILOSOPHY:
-------------------
We will understand normalization through:
1. MATHEMATICAL: The min-max transformation formula
2. GEOMETRIC: How it affects the data space
3. PRACTICAL: When to use it versus standardization

THE TRANSFORMATION:
------------------
For each feature j, we transform to range [min_val, max_val]:

X_scaled = (X - X_min) / (X_max - X_min)  # Scales to [0, 1]
X_scaled = X_scaled * (max_val - min_val) + min_val  # Shifts to desired range

Common choices:
- [0, 1]: Most common, interpretable as "fraction of total range"
- [-1, 1]: Symmetric around zero, useful for signed data
- [0, 255]: For image processing (pixel values)

STANDARDIZATION VS NORMALIZATION:
--------------------------------
STANDARDIZATION (Z-score):
- Centers at 0, scales by std
- Resulting range unbounded (typically -3 to +3, but can be larger)
- Preserves shape of distribution
- Robust to outliers if using median and IQR
- Best for: Gradient descent, neural networks, most ML algorithms

NORMALIZATION (Min-Max):
- Bounds to specific range [a, b]
- Preserves exact minimum and maximum
- Very sensitive to outliers (one extreme value affects everything!)
- Best for: Neural networks with bounded activations, image processing

REAL-WORLD ANALOGY:
------------------
Imagine grading exams where scores range from 45 to 95.

Normalization to [0, 1]:
- 45 (worst) → 0.0
- 70 (middle) → 0.5
- 95 (best) → 1.0
Everyone is scaled to "fraction of the range."

Standardization:
- Mean is 70, std is 15
- 45 → (45-70)/15 = -1.67 (below average)
- 70 → 0 (average)
- 95 → +1.67 (above average)
Everyone is measured in "deviations from average."

GEOMETRIC INTERPRETATION:
------------------------
Normalization squashes the data into a box with corners at [0, 0, ..., 0]
and [1, 1, ..., 1]. The shape is preserved, just scaled and shifted.

Before: Data might span [100, 5000] × [2, 8] (rectangular)
After: Data spans [0, 1] × [0, 1] (unit square)

THE OUTLIER PROBLEM:
-------------------
This is normalization's biggest weakness!

Example data: [1, 2, 3, 4, 5, 100]  (one outlier at 100)

After normalization to [0, 1]:
[0.00, 0.01, 0.02, 0.03, 0.04, 1.00]

The outlier pushed everything else to near zero! Five values are crammed
into [0, 0.04] while the outlier gets [0.04, 1.00] all to itself.

With standardization, the outlier has less dramatic effect because std is
more robust than range.

WHEN TO USE NORMALIZATION:
-------------------------
✓ Neural networks with bounded activation (sigmoid, tanh)
✓ Image data (pixel intensities naturally in [0, 255])
✓ When you need outputs in specific range
✓ When data has no outliers or outliers are meaningful
✓ When features already have bounded ranges

✗ Data with outliers
✗ Data where outliers need to be preserved
✗ When you want statistical properties (mean=0, std=1)

MACHINE LEARNING APPLICATIONS:
-----------------------------
1. NEURAL NETWORKS:
   Input normalization to [0, 1] works well with sigmoid: σ(x) = 1/(1+e^-x)
   Output naturally in [0, 1], matching normalized inputs.

2. IMAGE PROCESSING:
   Pixel values are naturally [0, 255]. Normalizing to [0, 1] makes them
   compatible with modern deep learning frameworks.

3. BOUNDED FEATURES:
   If features have natural bounds (probabilities, percentages), normalization
   preserves those bounds while scaling to a standard range.

4. DISTANCE METRICS:
   When computing distances, having all features in [0, 1] ensures equal
   contribution (like standardization but with bounded range).

Let's implement this carefully!
"""
from typing import Optional, Tuple

from foundations.linear_algebra.vectors_and_matrices import Matrix, Vector
from foundations.statistics.descriptive import *


class MinMaxScaler:
    """
    Transform features by scaling each feature to a given range.
    This implementation uses custom Vector and Matrix classes.
    """
    def __init__(self, feature_range: Tuple[float, float] = (0, 1)):
        if feature_range[0] >= feature_range[1]:
            raise ValueError("Minimum of feature_range must be smaller than maximum.")
        self.feature_range = feature_range
        self.data_min_: Optional[Vector] = None
        self.data_max_: Optional[Vector] = None
        self.scale_: Optional[Vector] = None
        self.min_: Optional[Vector] = None
        self.n_features_: Optional[int] = None

    def fit(self, X: Matrix) -> 'MinMaxScaler':
        """
        Compute the min/max for each feature to be used for later scaling.
        """
        self.n_features_ = X.num_cols
        
        # Find min and max for each feature (column)
        min_elements = [min(X.column(j).elements) for j in range(self.n_features_)]
        max_elements = [max(X.column(j).elements) for j in range(self.n_features_)]
        
        self.data_min_ = Vector(min_elements)
        self.data_max_ = Vector(max_elements)
        
        # Calculate the range for each feature
        data_range_vec = self.data_max_ - self.data_min_
        
        # Handle constant features (range = 0) to avoid division by zero
        for i in range(len(data_range_vec)):
            if data_range_vec[i] == 0:
                data_range_vec.elements[i] = 1.0

        # Precompute scale and min vectors for efficient transformation
        range_min, range_max = self.feature_range
        
        scale_elements = [(range_max - range_min) / dr for dr in data_range_vec.elements]
        self.scale_ = Vector(scale_elements)
        
        min_elements = [range_min - self.data_min_[i] * self.scale_[i] for i in range(self.n_features_)]
        self.min_ = Vector(min_elements)
        
        return self

    def transform(self, X: Matrix) -> Matrix:
        """
        Scale features of X according to the learned feature_range.
        """
        if self.scale_ is None or self.min_ is None:
            raise ValueError("MinMaxScaler has not been fitted yet. Call 'fit' first.")
        if X.num_cols != self.n_features_:
            raise ValueError(f"Input has {X.num_cols} features but scaler was fitted on {self.n_features_}")

        scaled_rows = []
        # Apply transformation element-wise: X_scaled = X * scale_ + min_
        for i in range(X.num_rows):
            new_row = [X[i, j] * self.scale_[j] + self.min_[j] for j in range(self.n_features_)]
            scaled_rows.append(new_row)
            
        return Matrix(scaled_rows)

    def fit_transform(self, X: Matrix) -> Matrix:
        """A convenience method to fit to data, then transform it."""
        return self.fit(X).transform(X)

    def inverse_transform(self, X_scaled: Matrix) -> Matrix:
        """Undo the scaling of X according to the learned feature_range."""
        if self.scale_ is None or self.min_ is None:
            raise ValueError("MinMaxScaler has not been fitted yet.")

        original_rows = []
        # Apply inverse transformation: X = (X_scaled - min_) / scale_
        for i in range(X_scaled.num_rows):
            new_row = [(X_scaled[i, j] - self.min_[j]) / self.scale_[j] for j in range(self.n_features_)]
            original_rows.append(new_row)
            
        return Matrix(original_rows)

# ==============================================================================
# UPDATED USAGE EXAMPLES AND TESTS
# ==============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("NORMALIZATION (MIN-MAX SCALING): INTERACTIVE TUTORIAL")
    print("=" * 70)
    
    print("\n### BASIC NORMALIZATION TO [0, 1] ###\n")
    
    X_train = Matrix([
        [1.0, 100.0],
        [2.0, 200.0],
        [3.0, 300.0],
        [4.0, 400.0],
        [5.0, 500.0]
    ])
    
    print("Original training data:")
    print(X_train)
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train_scaled = scaler.fit_transform(X_train)
    
    print("\nAfter normalization to [0, 1]:")
    print(X_train_scaled)
    print("→ Both features now in the [0, 1] range!")
    
    print("\n### THE OUTLIER PROBLEM ###\n")
    
    X_with_outlier = Matrix([
        [1.0, 10.0],
        [2.0, 12.0],
        [3.0, 11.0],
        [4.0, 13.0],
        [100.0, 15.0]  # Outlier in first feature!
    ])
    
    print("Data with outlier:")
    print(X_with_outlier)
    
    scaler_outlier = MinMaxScaler()
    X_outlier_scaled = scaler_outlier.fit_transform(X_with_outlier)
    
    print("\nAfter normalization:")
    print(X_outlier_scaled)
    print("\n→ Look at the first column: the outlier at 100 squashed all other values close to 0.")
    
    print("\n### TEST DATA OUTSIDE TRAINING RANGE ###\n")
    
    X_train_simple = Matrix([[1.0], [2.0], [3.0]])
    X_test_extreme = Matrix([[0.0], [2.5], [5.0]])
    
    scaler_test = MinMaxScaler()
    scaler_test.fit(X_train_simple)
    X_test_scaled = scaler_test.transform(X_test_extreme)
    
    print("Training data range: [1.0, 3.0]")
    print("Test data points: 0.0, 2.5, 5.0")
    print(f"\nNormalized test data:")
    print(X_test_scaled)
    print("→ Test values are outside [0, 1] because they exceed the training range!")

    print("\n### INVERSE TRANSFORMATION ###\n")
    
    X_reconstructed = scaler.inverse_transform(X_train_scaled)
    print("Original data:")
    print(X_train)
    print("\nReconstructed from normalized:")
    print(X_reconstructed)

    # A simple check for equality
    is_close = abs(X_train[0, 0] - X_reconstructed[0, 0]) < 1e-9
    print(f"\nPerfect reconstruction? (checking one element): {is_close}")
    
    print("\n✅ Normalization with custom classes complete!")