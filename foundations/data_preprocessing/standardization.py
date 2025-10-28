"""
STANDARDIZATION (Z-SCORE NORMALIZATION): MAKING FEATURES COMPARABLE
===================================================================

Welcome to the most important preprocessing technique in machine learning!

Standardization transforms each feature to have mean=0 and standard deviation=1.
This simple transformation solves one of the biggest problems in ML: features on
wildly different scales that confuse algorithms and prevent learning.

TEACHING PHILOSOPHY:
-------------------
We will understand standardization from three perspectives:
1. STATISTICAL: What it does to data distributions
2. GEOMETRIC: How it affects distances in feature space
3. PRACTICAL: Why algorithms need it and when to apply it

THE PROBLEM IT SOLVES:
----------------------
Imagine training a house price predictor with these features:
- Square footage: ranges from 500 to 5000 (scale of thousands)
- Number of bedrooms: ranges from 1 to 5 (scale of ones)
- Distance to city: ranges from 0 to 50 miles (scale of tens)

Without standardization:
- Gradient descent will be dominated by square footage (largest scale)
- Small changes to bedrooms barely affect the loss
- Learning will zig-zag slowly because gradients have very different magnitudes

With standardization:
- All features contribute equally to distance calculations
- Gradients have similar magnitudes across features
- Convergence is fast and smooth

REAL-WORLD ANALOGY:
------------------
Imagine comparing athletes using different metrics:
- Height in centimeters: 180, 175, 190
- Weight in kilograms: 75, 70, 85
- 100m sprint time in seconds: 11.5, 11.2, 11.8

These are on completely different scales! To compare athletes fairly, we standardize
each metric. Then we can say "this athlete is 1.5 standard deviations above average
in height, 0.5 below average in weight, and 0.8 above average in speed."

THE MATHEMATICAL TRANSFORMATION:
-------------------------------
For each feature j:
1. Compute mean: μ_j = (1/n) Σ x_ij
2. Compute standard deviation: σ_j = sqrt((1/n) Σ (x_ij - μ_j)²)
3. Transform: z_ij = (x_ij - μ_j) / σ_j

After transformation:
- Mean of z_j = 0 (centered at origin)
- Standard deviation of z_j = 1 (unit variance)

WHY THIS WORKS:
--------------
Subtracting the mean centers the data at zero. Dividing by standard deviation
scales it so one unit represents "one standard deviation from the mean."

Now -1.5 means "1.5 standard deviations below average" regardless of the original
units. This makes features directly comparable!

GEOMETRIC INTERPRETATION:
------------------------
In feature space, standardization makes the data cloud more spherical.

Before standardization: elongated ellipsoid (stretched along high-variance features)
After standardization: more spherical (equal spread in all directions)

This spherical shape makes distance metrics meaningful and gradient descent efficient.

WHEN TO USE:
-----------
ALWAYS use standardization for:
- Linear regression (especially with gradient descent)
- Logistic regression
- Neural networks
- Support Vector Machines
- K-Nearest Neighbors
- Any distance-based algorithm

DO NOT standardize for:
- Decision trees and random forests (they're scale-invariant)
- Naive Bayes (works with counts/probabilities directly)

CRITICAL RULES:
--------------
1. FIT on training data ONLY: Learn mean and std from training set
2. TRANSFORM both train and test using training statistics
3. NEVER fit on test data (that's data leakage!)

Example of correct workflow:
```python
scaler = Standardizer()
scaler.fit(X_train)  # Learn mean and std from training data
X_train_scaled = scaler.transform(X_train)  # Apply to training
X_test_scaled = scaler.transform(X_test)   # Apply same transformation to test
```

MACHINE LEARNING IMPACT:
-----------------------
Without standardization:
- Gradient descent: slow convergence, zig-zagging
- Neural networks: unstable training, vanishing/exploding gradients
- KNN: dominated by large-scale features
- Regularization: unfairly penalizes small-scale features

With standardization:
- Gradient descent: fast, smooth convergence
- Neural networks: stable training
- KNN: all features contribute equally
- Regularization: fair penalty across features

Now Let's implement this crucial transformation!
"""

import numpy as np
from typing import Optional, Tuple