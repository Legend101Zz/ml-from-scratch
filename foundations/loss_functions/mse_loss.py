"""
MEAN SQUARED ERROR (MSE): THE FOUNDATION OF REGRESSION
======================================================

MSE is the most fundamental loss function in machine learning. Understanding MSE deeply
means understanding why we minimize losses, how gradients guide learning, and the
mathematical foundation connecting statistics to optimization.

TEACHING PHILOSOPHY:
-------------------
We'll explore MSE from five complementary perspectives:
1. GEOMETRIC: Measuring distance between predictions and truth
2. STATISTICAL: Connection to Gaussian noise and maximum likelihood
3. CALCULUS: Why squared errors give clean gradients
4. PRACTICAL: When to use MSE vs alternatives
5. PHILOSOPHICAL: What "best fit" really means

THE MATHEMATICAL DEFINITION:
---------------------------
For predictions ŷ and true values y:

MSE = (1/n) Σ(ŷᵢ - yᵢ)²

Where:
- n is the number of samples
- ŷᵢ is the predicted value for sample i
- yᵢ is the true value for sample i
- (ŷᵢ - yᵢ) is the error (residual) for sample i

We square each error, then average across all samples.

WHY SQUARED ERRORS?
------------------
Four compelling reasons:

**1. GEOMETRIC INTERPRETATION: Euclidean Distance**
MSE measures squared L2 distance between prediction and target vectors.
In n-dimensional space, the best prediction is the point closest to the target.
Squaring ensures distance is always positive and emphasizes large errors.

**2. STATISTICAL INTERPRETATION: Maximum Likelihood**
If errors are Gaussian (normal) distributed: ε ~ N(0, σ²)
Then maximizing likelihood is EXACTLY equivalent to minimizing MSE!

Proof:
P(y|x) = (1/√(2πσ²)) exp(-(y - ŷ)²/(2σ²))
Log-likelihood: log P(y|x) = -(y - ŷ)²/(2σ²) + constant
Maximizing log-likelihood = Minimizing (y - ŷ)²

This is profound: MSE is the "natural" loss function when noise is Gaussian!

**3. MATHEMATICAL CONVENIENCE: Clean Gradients**
The derivative of x² is 2x (simple!).
∂MSE/∂ŷ = (2/n) Σ(ŷᵢ - yᵢ)

This clean gradient makes optimization algorithms (gradient descent) very efficient.
Compare to absolute error |x| which has discontinuous gradient at x=0.

**4. PENALIZES LARGE ERRORS HEAVILY**
Small error: 0.1² = 0.01
Large error: 10² = 100

The large error contributes 10,000x more to the loss!
This forces the model to avoid large mistakes, even if it means making more small ones.

RMSE (ROOT MEAN SQUARED ERROR):
-------------------------------
RMSE = √MSE

Why take the square root?
- Returns error to original units (MSE has squared units)
- More interpretable: RMSE of $1000 means "typical error is $1000"
- Still emphasizes large errors (via the squaring before rooting)

For optimization, MSE and RMSE are equivalent (same minimum).
For reporting, RMSE is more interpretable.

MSE VS MAE (MEAN ABSOLUTE ERROR):
---------------------------------
MSE = (1/n) Σ(ŷᵢ - yᵢ)²     (Square errors)
MAE = (1/n) Σ|ŷᵢ - yᵢ|      (Absolute errors)

**MSE Characteristics:**
✓ Heavily penalizes outliers (good if outliers are errors)
✓ Smooth gradient (easy optimization)
✓ Assumes Gaussian noise (maximum likelihood)
✗ Sensitive to outliers (bad if outliers are meaningful)
✗ Not robust (one huge error dominates loss)

**MAE Characteristics:**
✓ Robust to outliers (treats all errors equally)
✓ Corresponds to median prediction (not mean)
✗ Non-smooth gradient at zero (harder optimization)
✗ Assumes Laplacian noise (less common)

**When to use which:**
- MSE: Normal noise, no outliers, want smooth optimization
- MAE: Heavy-tailed noise, outliers are meaningful, want robust model
- Huber Loss: Compromise (MSE for small errors, MAE for large)

THE BIAS-VARIANCE TRADEOFF:
--------------------------
MSE can be decomposed into three components:

MSE = Bias² + Variance + Irreducible Error

**Bias**: Systematic error from wrong model assumptions
- High bias: model too simple, underfits

**Variance**: Error from sensitivity to training data fluctuations
- High variance: model too complex, overfits

**Irreducible Error**: Noise in the data itself
- Cannot be reduced (fundamental limit)

This decomposition explains the tradeoff:
- Complex model: Low bias, high variance
- Simple model: High bias, low variance
- Goal: Find sweet spot minimizing total MSE

GRADIENT FOR LEARNING:
---------------------
The gradient of MSE with respect to predictions:

∂MSE/∂ŷ = (2/n) Σ(ŷᵢ - yᵢ) = (2/n)(ŷ - y)

For linear regression (ŷ = Xw):
∂MSE/∂w = (2/n)Xᵀ(Xw - y)

This gradient points "uphill" in loss. To minimize loss, we move opposite to gradient:
w_new = w_old - learning_rate * gradient

This is gradient descent! The foundation of all modern ML optimization.

WHEN MSE FAILS:
--------------

**1. Outliers dominate**
Data: [1.0, 1.1, 0.9, 1.2, 100.0]  (one huge outlier)
MSE heavily weights the outlier, model chases it
Solution: Use MAE or robust regression

**2. Heteroscedastic noise** (variance depends on x)
Example: Predicting income (high earners have more variance)
MSE assumes constant variance
Solution: Use weighted MSE or quantile regression

**3. Asymmetric costs**
Example: Predicting hospital stay (underestimating is worse than overestimating)
MSE treats both equally
Solution: Use asymmetric loss or custom loss function

**4. Non-Gaussian errors**
If errors follow heavy-tailed distribution (Cauchy, Student-t)
MSE is not maximum likelihood
Solution: Use loss matching the error distribution

MACHINE LEARNING APPLICATIONS:
-----------------------------

**1. LINEAR REGRESSION**
Minimize MSE to find best-fit line
Closed-form solution: w = (XᵀX)⁻¹Xᵀy
Or iterative: gradient descent

**2. NEURAL NETWORKS (Regression)**
Output layer: Linear activation
Loss function: MSE
Backpropagation computes gradients through entire network

**3. AUTOENCODER**
Reconstruct input: minimize MSE between input and reconstruction
Forces network to learn compressed representation

**4. GENERATIVE MODELS**
Some GANs use MSE to match distributions
Pixel-level image reconstruction uses MSE

**5. REINFORCEMENT LEARNING**
Value function learning: minimize MSE between predicted and true returns
Q-learning updates based on temporal difference (a form of MSE)

Let's implement MSE with all its variations!
"""


import math
from typing import Any, Dict, List, Tuple

from foundations.linear_algebra.vectors_and_matrices import Matrix, Vector

# ==============================================================================
#  MSE LOSS FUNCTIONS
# ==============================================================================

def mean_squared_error(y_true: Matrix, y_pred: Matrix) -> float:
    """
    Compute the mean squared error between true and predicted values using Matrix.
    """
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true has shape {y_true.shape}, "
            f"y_pred has shape {y_pred.shape}"
        )
    
    # Calculate (y_pred - y_true)
    errors_matrix = y_pred - y_true
    
    # Square each error element-wise
    squared_errors_matrix = errors_matrix.power(2)
    
    # Average over all elements
    mse = squared_errors_matrix.mean_all()
    
    return float(mse)


def root_mean_squared_error(y_true: Matrix, y_pred: Matrix) -> float:
    """
    Compute the root mean squared error (RMSE) using Matrix.
    """
    mse = mean_squared_error(y_true, y_pred)
    return math.sqrt(mse)


def mse_gradient(y_true: Matrix, y_pred: Matrix) -> Matrix:
    """
    Compute the gradient of MSE with respect to predictions, using Matrix.
    
    ∂MSE/∂y_pred = (2/n) * (y_pred - y_true)
    """
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true has shape {y_true.shape}, "
            f"y_pred has shape {y_pred.shape}"
        )
    
    n_samples = y_true.num_rows * y_true.num_cols # Total number of elements
    
    # Calculate (y_pred - y_true)
    errors_matrix = y_pred - y_true
    
    # Scale by (2/n)
    gradient_matrix = errors_matrix * (2.0 / n_samples)
    
    return gradient_matrix


def r2_score(y_true: Matrix, y_pred: Matrix) -> float:
    """
    Compute R² (coefficient of determination) regression score using Matrix.
    """
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true has shape {y_true.shape}, "
            f"y_pred has shape {y_pred.shape}"
        )

    # Mean of true values (flatten to get a single mean)
    y_true_flat = y_true.flatten()
    if not y_true_flat:
        return 0.0 # Handle empty matrix case
    y_mean = sum(y_true_flat) / len(y_true_flat)
    
    # Total sum of squares (variance in data)
    ss_tot = 0.0
    for val in y_true_flat:
        ss_tot += (val - y_mean) ** 2
    
    # Residual sum of squares (unexplained variance)
    ss_res = 0.0
    for r in range(y_true.num_rows):
        for c in range(y_true.num_cols):
            ss_res += (y_true.elements[r][c] - y_pred.elements[r][c]) ** 2
            
    if ss_tot == 0:
        # If y_true is constant, R2 is undefined.
        # Common behavior is 1.0 if predictions are also constant and correct, else 0.0
        return 1.0 if ss_res == 0 else 0.0

    # R² score
    r2 = 1 - (ss_res / ss_tot)
    
    return float(r2)


# ==============================================================================
# USAGE EXAMPLES AND TESTS
# ==============================================================================

# Helper function to generate Matrix data for examples, replacing np.array
def create_matrix(data: List[List[Any]]) -> Matrix:
    # Ensure inner lists are handled for single-row inputs as well
    if not data or not isinstance(data[0], list):
        return Matrix([data])
    return Matrix(data)

if __name__ == "__main__":
    print("=" * 70)
    print("MEAN SQUARED ERROR (MSE): INTERACTIVE TUTORIAL (using custom Matrix)")
    print("=" * 70)
    
    print("\n### BASIC MSE COMPUTATION ###\n")
    
    y_true_data = [[3.0], [-0.5], [2.0], [7.0]]
    y_pred_data = [[2.5], [0.0], [2.0], [8.0]]
    
    y_true = create_matrix(y_true_data)
    y_pred = create_matrix(y_pred_data)
    
    print("True values:", y_true)
    print("Predictions:", y_pred)
    
    # Calculate errors manually for display
    errors_display = y_pred - y_true
    print("Errors (pred - true):", errors_display)
    
    mse = mean_squared_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    
    print(f"\nMSE = {mse:.4f}")
    print(f"RMSE = {rmse:.4f}")
    
    # Manual calculation for verification
    squared_errors_display = errors_display.power(2)
    print(f"\nSquared errors: {squared_errors_display}")
    print(f"Mean of squared errors: {squared_errors_display.mean_all():.4f}")
    print("→ Matches our MSE ✓")
    
    print("\n### WHY SQUARED: PENALIZING LARGE ERRORS ###\n")
    
    small_error = 0.5
    large_error = 5.0
    
    print(f"Small error: {small_error} → squared: {small_error**2}")
    print(f"Large error: {large_error} → squared: {large_error**2}")
    print(f"Ratio: {(large_error**2) / (small_error**2):.1f}x")
    print("\n→ Large error contributes 100x more to MSE!")
    print("  This forces model to avoid big mistakes")
    
    print("\n### MSE GRADIENT ###\n")
    
    grad = mse_gradient(y_true, y_pred)
    print(f"Gradient: {grad}")
    print("\nInterpretation:")
    for i in range(grad.num_rows):
        val_grad = grad.elements[i][0]
        if val_grad > 0:
            print(f"  Sample {i}: grad={val_grad:.4f} → prediction too high, decrease it")
        elif val_grad < 0:
            print(f"  Sample {i}: grad={val_grad:.4f} → prediction too low, increase it")
        else:
            print(f"  Sample {i}: grad={val_grad:.4f} → perfect! no change needed")
    
    print("\n### R² SCORE: EXPLAINED VARIANCE ###\n")
    
    # Perfect predictions
    y_perfect = create_matrix(y_true_data) # Create a copy
    r2_perfect = r2_score(y_true, y_perfect)
    print(f"Perfect predictions: R² = {r2_perfect:.4f}")
    
    # Mean prediction (useless model)
    y_true_flat_for_mean = y_true.flatten()
    mean_val_for_pred = sum(y_true_flat_for_mean) / len(y_true_flat_for_mean)
    y_mean_pred_data = [[mean_val_for_pred] for _ in y_true_data]
    y_mean_pred = create_matrix(y_mean_pred_data)
    r2_mean = r2_score(y_true, y_mean_pred)
    print(f"Always predict mean: R² = {r2_mean:.4f}")
    
    # Our actual predictions
    r2_actual = r2_score(y_true, y_pred)
    print(f"Our predictions: R² = {r2_actual:.4f}")
    print(f"\n→ Model explains {r2_actual*100:.2f}% of variance!")
    
    print("\n### MSE VS MAE: OUTLIER SENSITIVITY ###\n")
    
    # Data with no outliers
    y_normal = create_matrix([[1], [2], [3], [4], [5]])
    y_pred_normal = create_matrix([[1.1], [2.1], [2.9], [4.1], [4.9]])
    
    # Data with one outlier
    y_outlier = create_matrix([[1], [2], [3], [4], [100]])  # 100 is outlier!
    y_pred_outlier = create_matrix([[1.1], [2.1], [2.9], [4.1], [50]])  # Prediction also off
    
    mse_normal = mean_squared_error(y_normal, y_pred_normal)
    mse_outlier = mean_squared_error(y_outlier, y_pred_outlier)
    
    print("Normal data (no outliers):")
    print(f"  MSE = {mse_normal:.4f}")
    print("\nData with outlier:")
    print(f"  MSE = {mse_outlier:.4f}")
    print(f"\n→ Outlier inflates MSE by {mse_outlier/mse_normal:.1f}x!")
    print("  One bad prediction dominates the loss")
    print("  This is why MSE is sensitive to outliers")
    
    print("\n### COMPARING MODELS WITH MSE ###\n")
    
    # Simulating data generation with `random` and `math`
    # True data (sine wave with noise)
    random.seed(42)
    X_vals = [i * 10 / 49 for i in range(50)] # Equivalent to np.linspace(0, 10, 50)
    y_true_sine_data = [[math.sin(x) + random.gauss(0, 0.1)] for x in X_vals]
    y_true_sine = create_matrix(y_true_sine_data)
    
    # Model 1: Predict mean (baseline)
    y_true_sine_flat = y_true_sine.flatten()
    mean_sine = sum(y_true_sine_flat) / len(y_true_sine_flat)
    y_pred_mean_data = [[mean_sine] for _ in range(y_true_sine.num_rows)]
    y_pred_mean = create_matrix(y_pred_mean_data)
    
    # Model 2: Linear fit (simplified for custom Matrix without full regression)
    # We'll simulate a 'better' linear prediction for demonstration, not truly fit
    # In a real scenario, you'd implement linear regression first
    y_pred_linear_data = [[0.1 * x - 0.2 + random.gauss(0, 0.05)] for x in X_vals] # Simplified example
    y_pred_linear = create_matrix(y_pred_linear_data)
    
    # Model 3: Polynomial fit (simplified - just a generally good fit)
    y_pred_poly_data = [[math.sin(x) + random.gauss(0, 0.02)] for x in X_vals] # Near perfect prediction
    y_pred_poly = create_matrix(y_pred_poly_data)
    
    mse_mean = mean_squared_error(y_true_sine, y_pred_mean)
    mse_linear = mean_squared_error(y_true_sine, y_pred_linear)
    mse_poly = mean_squared_error(y_true_sine, y_pred_poly)
    
    print(f"Always predict mean: MSE = {mse_mean:.4f}")
    print(f"Linear model (simulated): MSE = {mse_linear:.4f}")
    print(f"Polynomial model (simulated): MSE = {mse_poly:.4f}")
    print("\n→ Lower MSE = better fit")
    print("  Polynomial captures sine wave pattern best (simulated perfect fit)")
    
    print("\n### KEY TAKEAWAYS ###\n")
    print("""
1. MSE = average of squared errors
   - Always non-negative
   - Zero means perfect fit
   - Large values mean bad predictions

2. Squared errors penalize large mistakes heavily
   - Forces model to avoid big errors
   - One outlier can dominate the loss

3. RMSE puts error back in original units
   - More interpretable than MSE
   - Represents "typical" error magnitude

4. R² measures explained variance
   - 1.0 = perfect, 0.0 = no better than mean
   - Scale-independent (compare across problems)

5. MSE assumes Gaussian noise
   - Maximum likelihood under normality
   - Use MAE if outliers are meaningful
    """)
    
    print("\n✅ MSE loss (with custom Matrix) complete!")