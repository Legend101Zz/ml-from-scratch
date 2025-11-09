"""
TRAIN-TEST SPLIT: THE FOUNDATION OF HONEST EVALUATION
=====================================================

The single most important rule in machine learning: NEVER test on training data!

This simple principle separates real machine learning from self-deception. Train-test
splitting is how we honestly evaluate whether our model has learned general patterns
or just memorized the training data.

TEACHING PHILOSOPHY:
-------------------
We'll understand train-test splitting from multiple perspectives:
1. PHILOSOPHICAL: Why honest evaluation matters
2. MATHEMATICAL: How to split randomly vs strategically
3. PRACTICAL: Common mistakes and how to avoid them

THE FUNDAMENTAL PROBLEM: OVERFITTING
------------------------------------
Imagine a student who memorizes answers to practice problems without understanding
the concepts. They score 100% on practice but fail the real exam.

This is exactly what happens without train-test split:
- Model memorizes training data → perfect training accuracy
- Model fails on new data → poor test accuracy
- You think you have a great model, but it's useless in production!

Train-test split solves this by holding out a test set the model NEVER sees during
training. If the model performs well on this unseen data, it has truly learned!

MATHEMATICAL FORMULATION:
------------------------
Given dataset D with n samples:
1. Randomly split into training set D_train and test set D_test
2. Train model only on D_train
3. Evaluate model on D_test (honest performance estimate)

Common split ratios:
- 80/20: 80% train, 20% test (most common)
- 70/30: More test data for better performance estimate
- 90/10: When data is limited (less reliable test estimate)

The split percentage depends on:
- Total data size (more data → can afford larger test set)
- Model complexity (complex models need more training data)
- Variability (high variance → need larger test set for stable estimate)

RANDOM VS STRATIFIED SPLITTING:
-------------------------------
**RANDOM SPLIT**: Each sample has equal probability of being in test set
- Pro: Simple, unbiased
- Con: Might create imbalanced splits for rare classes

**STRATIFIED SPLIT**: Preserve class distribution in both sets
- Pro: Both sets representative of overall distribution
- Con: Only applicable to classification, requires class labels

Example:
Dataset: 90 cats, 10 dogs (90% cats, 10% dogs)

Random split (bad luck):
- Train: 75 cats, 5 dogs (93.75% cats, 6.25% dogs)
- Test: 15 cats, 5 dogs (75% cats, 25% dogs)
→ Test set has 2.5x more dogs than training! Model hasn't seen enough dogs!

Stratified split:
- Train: 72 cats, 8 dogs (90% cats, 10% dogs)
- Test: 18 cats, 2 dogs (90% cats, 10% dogs)
→ Both sets maintain the 90/10 distribution ✓

SHUFFLING: BREAKING TEMPORAL CORRELATIONS
------------------------------------------
Always shuffle before splitting UNLESS your data has temporal structure!

**When to shuffle**: Independent samples (images, text documents, user profiles)
- Ensures no hidden ordering affects the split
- Prevents first 80% vs last 20% having different distributions

**When NOT to shuffle**: Time series data (stock prices, weather, user activity over time)
- Must respect temporal order!
- Train on past, test on future
- Shuffling would leak future information into training (cheating!)

Example:
Stock prices: [Jan, Feb, Mar, Apr, May, Jun]
✗ Random split: Train on [Jan, Apr, Jun], Test on [Feb, Mar, May]
  → Training on future data! Model sees the future!
✓ Sequential split: Train on [Jan, Feb, Mar, Apr], Test on [May, Jun]
  → Realistic: model predicts future from past

RANDOM SEED: REPRODUCIBILITY
----------------------------
Setting a random seed ensures the same split across runs:
- Important for comparing models fairly
- Enables reproducibility in research
- Allows debugging (same data splits every time)

Without seed: Different split every run → can't compare results!
With seed: Same split every run → fair comparison

DATA LEAKAGE: THE SILENT KILLER
-------------------------------
Data leakage is when information from test set influences training. It makes
performance estimates overly optimistic and models fail in production.

Common leakage sources:

**1. Preprocessing before splitting**
❌ Wrong:
```python
X_normalized = normalize(X)  # Uses stats from ALL data!
X_train, X_test = split(X_normalized)
```
Why wrong: Test set statistics influenced training normalization!

✓ Correct:
```python
X_train, X_test = split(X)
scaler.fit(X_train)  # Learn stats from training only
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)  # Apply training stats to test
```

**2. Feature selection on full dataset**
❌ Wrong: Select features using all data, then split
✓ Correct: Split first, select features using only training data

**3. Hyperparameter tuning on test set**
❌ Wrong: Try different parameters, pick best on test set
✓ Correct: Use validation set or cross-validation

**4. Temporal leakage in time series**
❌ Wrong: Train on random samples including future data
✓ Correct: Train only on past, test on future

VALIDATION SET: TUNING WITHOUT PEEKING
--------------------------------------
Problem: We need test set for honest evaluation, but also need to tune hyperparameters.
Solution: Split into THREE sets!

Training set: Learn model parameters (60%)
Validation set: Tune hyperparameters (20%)
Test set: Final honest evaluation (20%)

Workflow:
1. Split: train/val/test
2. Train multiple models with different hyperparameters on train set
3. Evaluate all models on validation set
4. Pick best model based on validation performance
5. ONLY THEN evaluate once on test set (honest estimate)

Test set is sacred: touch it once, at the very end!

CROSS-VALIDATION: USING DATA EFFICIENTLY
----------------------------------------
When data is limited, train-test split wastes data (test set not used for training).
Cross-validation uses all data for both training and testing!

K-Fold Cross-Validation:
1. Split data into K folds
2. For each fold i:
   - Train on all folds except i
   - Test on fold i
3. Average performance across all K tests

This gives K different models and performance estimates, using 100% of data.

Still need final test set? Yes! Use cross-validation to tune hyperparameters,
then evaluate final model on held-out test set.

REGRESSION VS CLASSIFICATION:
-----------------------------
**Regression**: Just split randomly (or stratify on binned target values)
**Classification**: Always use stratified split to preserve class balance!

For multi-class: Stratify on the class with smallest representation.

MACHINE LEARNING PIPELINE:
--------------------------
Correct order:
1. Split data (train/test or train/val/test)
2. Fit preprocessing on training data
3. Transform training data
4. Transform test data using training statistics
5. Train model on transformed training data
6. Evaluate on transformed test data

NEVER:
- Fit preprocessing on test data
- Use test data for any decision (feature selection, hyperparameters)
- Look at test performance during development (only at the very end!)

Let's implement train-test split properly!
"""

import random
from typing import Any, Dict, List, Optional, Tuple, Union

from foundations.linear_algebra.vectors_and_matrices import Matrix, Vector
from foundations.statistics.descriptive import *


def _stratified_split(y: Matrix,
                      n_samples: int,
                      n_test: int,
                      random_state: Optional[int]
                      ) -> Tuple[List[int], List[int]]:
    """
    Perform stratified split to maintain class distribution, using only Python.
    """
    if random_state is not None:
        random.seed(random_state)
        
    y_labels = y.column(0) # Assume y is a single-column Matrix
    
    # 1. Find all unique classes and their indices
    class_indices: Dict[Any, List[int]] = {}
    for i, label in enumerate(y_labels):
        if label not in class_indices:
            class_indices[label] = []
        class_indices[label].append(i)
        
    train_idx: List[int] = []
    test_idx: List[int] = []
    
    # 2. For each class, compute how many samples should be in test set
    for class_label, indices in class_indices.items():
        n_class = len(indices)
        
        # 3. Compute number of test samples for this class
        n_class_test = int(round(n_test * n_class / n_samples))
        # Ensure at least 1 in each set if possible
        n_class_test = max(0, min(n_class_test, n_class))
        if n_class - n_class_test < 1 and n_class > 1:
            n_class_test = n_class - 1 # Keep at least one for training
        if n_class_test < 1 and n_class > 1:
             n_class_test = 1 # Keep at least one for testing
        
        # 4. Randomly sample that many indices from each class
        random.shuffle(indices)
        
        test_idx.extend(indices[:n_class_test])
        train_idx.extend(indices[n_class_test:])
        
    # 5. Shuffle the final lists to mix classes
    random.shuffle(train_idx)
    random.shuffle(test_idx)
    
    return train_idx, test_idx

def train_test_split(*arrays: Matrix,
                     test_size: Optional[Union[float, int]] = None,
                     train_size: Optional[Union[float, int]] = None,
                     random_state: Optional[int] = None,
                     shuffle: bool = True,
                     stratify: Optional[Matrix] = None
                     ) -> List[Matrix]:
    """
    Split Matrix arrays into random train and test subsets using pure Python.
    """
    if len(arrays) == 0:
        raise ValueError("At least one array required as input")
        
    # Check all arrays have same length
    n_samples = arrays[0].num_rows
    for arr in arrays[1:]:
        if arr.num_rows != n_samples:
            raise ValueError(f"All arrays must have same number of samples. Got {n_samples} and {arr.num_rows}")
            
    # Determine train and test sizes
    if test_size is None and train_size is None:
        test_size = 0.25
    
    if test_size is not None:
        if isinstance(test_size, float):
            if not 0.0 < test_size < 1.0:
                raise ValueError(f"test_size must be between 0 and 1, got {test_size}")
            n_test = int(n_samples * test_size)
        else:
            n_test = int(test_size)
    else: # test_size is None, compute from train_size
        if isinstance(train_size, float):
            if not 0.0 < train_size < 1.0:
                raise ValueError(f"train_size must be between 0 and 1, got {train_size}")
            n_train = int(n_samples * train_size)
        else:
            n_train = int(train_size)
        n_test = n_samples - n_train
        
    n_train = n_samples - n_test
    
    if n_train <= 0 or n_test <= 0:
        raise ValueError(f"Resulting train size {n_train} or test size {n_test} is invalid.")
        
    # Generate indices
    if stratify is not None:
        if stratify.num_rows != n_samples:
             raise ValueError("stratify matrix must have same number of samples as input arrays")
        train_idx, test_idx = _stratified_split(
            stratify, n_samples, n_test, random_state
        )
    else:
        # Regular split
        indices = list(range(n_samples))
        
        if shuffle:
            if random_state is not None:
                random.seed(random_state)
            random.shuffle(indices)
            
        test_idx = indices[:n_test]
        train_idx = indices[n_test:]
        
    # Split all arrays using the indices
    result: List[Matrix] = []
    for arr in arrays:
        # Manually build new matrices from selected rows
        train_rows = [arr.row(i) for i in train_idx]
        test_rows = [arr.row(i) for i in test_idx]
        
        result.append(Matrix(train_rows))
        result.append(Matrix(test_rows))
        
    return result

# ==============================================================================
# UPDATED USAGE EXAMPLES AND TESTS
# ==============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("TRAIN-TEST SPLIT: INTERACTIVE TUTORIAL (using custom Matrix)")
    print("=" * 70)
    
    print("\n### BASIC SPLIT ###\n")
    
    # Create simple dataset
    X_data = [[i, i+1] for i in range(0, 20, 2)]
    y_data = [[i % 2] for i in range(10)]
    X = Matrix(X_data)
    y = Matrix(y_data)
    
    print("Original data:")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    print(f"\nAfter 70/30 split:")
    print(f"Training set: X_train shape {X_train.shape}, y_train shape {y_train.shape}")
    print(f"Test set: X_test shape {X_test.shape}, y_test shape {y_test.shape}")
    
    print("\n### REPRODUCIBILITY WITH RANDOM STATE ###\n")
    
    # Same split multiple times
    X_train1, X_test1, _, _ = train_test_split(X, y, test_size=0.3, random_state=42)
    X_train2, X_test2, _, _ = train_test_split(X, y, test_size=0.3, random_state=42)
    
    print("Split 1 test X data:", X_test1.elements)
    print("Split 2 test X data:", X_test2.elements)
    print(f"Same split? {X_test1.elements == X_test2.elements} ✓")
    
    # Different random state
    X_train3, X_test3, _, _ = train_test_split(X, y, test_size=0.3, random_state=99)
    print(f"\nWith different random_state: {X_test1.elements == X_test3.elements}")
    print("→ Different split, as expected")
    
    print("\n### STRATIFIED SPLIT (IMPORTANT FOR CLASSIFICATION!) ###\n")
    
    # Imbalanced dataset: 80% class 0, 20% class 1
    X_imb_data = [[i] for i in range(50)]
    y_imb_data = [[0]]*40 + [[1]]*10
    X_imbalanced = Matrix(X_imb_data)
    y_imbalanced = Matrix(y_imb_data)
    
    def get_class_dist(y_matrix: Matrix) -> str:
        """Helper to count class occurrences in a Matrix"""
        counts: Dict[Any, int] = {}
        for row in y_matrix.elements:
            label = row[0]
            counts[label] = counts.get(label, 0) + 1
        total = y_matrix.num_rows
        if total == 0: return "Empty set"
        return ", ".join([f"Class {k}: {v}/{total} ({v/total*100:.1f}%)" for k, v in sorted(counts.items())])

    print(f"Original class distribution: {get_class_dist(y_imbalanced)}")
    
    # Split WITHOUT stratification
    _, _, _, y_test_random = train_test_split(
        X_imbalanced, y_imbalanced, test_size=0.2, random_state=42, stratify=None
    )
    
    print(f"\nWithout stratification (test set): {get_class_dist(y_test_random)}")
    print("→ Distribution might be different from original!")
    
    # Split WITH stratification
    _, _, _, y_test_stratified = train_test_split(
        X_imbalanced, y_imbalanced, test_size=0.2, random_state=42, stratify=y_imbalanced
    )
    
    print(f"\nWith stratification (test set):    {get_class_dist(y_test_stratified)}")
    print("→ Distribution matches original! ✓")
    
    print("\n### TIME SERIES: NO SHUFFLING! ###\n")
    
    # Time series data (e.g., stock prices by day)
    time_series_X_data = [[i] for i in range(1, 11)]
    dates = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct']
    dates_data = [[d] for d in dates]
    
    time_series_X = Matrix(time_series_X_data)
    dates_Matrix = Matrix(dates_data)

    print("Time series data (months):", dates)
    
    # WRONG: Shuffle time series
    _, X_test_wrong, _, y_test_wrong = train_test_split(
        time_series_X, dates_Matrix, test_size=0.3, random_state=42, shuffle=True
    )
    print(f"\n❌ With shuffle=True:")
    print(f"   Test on months: {[row[0] for row in y_test_wrong.elements]}")
    print("   → Training on future data! This is cheating!")
    
    # CORRECT: No shuffling for time series
    _, X_test_correct, _, y_test_correct = train_test_split(
        time_series_X, dates_Matrix, test_size=0.3, shuffle=False
    )
    print(f"\n✓ With shuffle=False:")
    print(f"   Test on months: {[row[0] for row in y_test_correct.elements]}")
    print("   → Train on past, test on future. Realistic! ✓")
    
    print("\n✅ Train-test split (with custom Matrix) complete!")